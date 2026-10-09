import type { Response } from 'express';
import type { SourceRegistry } from '../../utilt/sourceRegistry.js';
import {
    buildCitationLink,
    CITATION_PATTERN,
    CITATION_PREFIX,
    endsWithReferencesHeading,
    MAX_CITATION_LENGTH,
    REFERENCE_PLACEHOLDER,
    SOURCE_PREFIX,
} from '../../utilt/citation.js';

/** 与前端 fetchChatStream 兼容的增量载荷（content 走正文，reasoning 走思考/进度折叠区） */
export type SseDelta = { content?: string; reasoning?: string };

export type ToolCallChunkInfo = { id: string; name: string; args: string; index?: number };

/** 需要整体回退等待的标记前缀：任一标记到达前都不能把半截内容下发到前端 */
const HOLD_TOKENS = [CITATION_PREFIX, SOURCE_PREFIX, REFERENCE_PLACEHOLDER] as const;

/** 就地替换占位符时用于判断「References 标题是否已被模型输出」的已下发文本尾部保留长度 */
const EMITTED_TAIL_CHARS = 120;

/**
 * 写入一帧 SSE。
 * compression 中间件会缓冲 res.write()，必须显式 flush，否则流式会退化成一次性返回。
 */
export const writeSse = (res: Response, payload: string): void => {
    if (res.writableEnded) return;
    try {
        res.write(payload);
        const flush = (res as unknown as { flush?: () => void }).flush;
        flush?.();
    } catch {
        /* 客户端断开时忽略写入错误 */
    }
};

const buildDeltaPayload = (delta: SseDelta): Record<string, string> => {
    const payload: Record<string, string> = {};
    if (delta.content) payload.content = delta.content;
    if (delta.reasoning) payload.reasoning_content = delta.reasoning;
    return payload;
};

export const sseDeltaFrame = (delta: SseDelta): string => {
    const payload = buildDeltaPayload(delta);
    if (Object.keys(payload).length === 0) return '';
    return `data: ${JSON.stringify({ choices: [{ delta: payload }] })}\n\n`;
};

export const sseFinishFrame = (): string =>
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n`;

export const sseDoneFrame = (): string => 'data: [DONE]\n\n';

export const sseErrorFrame = (message: string): string =>
    `data: ${JSON.stringify({ error: { message } })}\n\n`;

const getMessageType = (message: unknown): string => {
    const getType = (message as { _getType?: () => string } | null)?._getType;
    return typeof getType === 'function' ? getType.call(message) : '';
};

export const isToolMessage = (message: unknown): boolean => getMessageType(message) === 'tool';

export const isAiMessage = (message: unknown): boolean => getMessageType(message) === 'ai';

/**
 * 提取推理增量。
 * 实测网关（DashScope 兼容模式 + deepseek 系模型）把 reasoning_content 放在
 * additional_kwargs 而不是标准 contentBlocks，这里优先取 additional_kwargs 并兼容 content 数组。
 */
export const extractReasoning = (chunk: unknown): string => {
    let reasoning = '';
    const kwargs = (chunk as { additional_kwargs?: Record<string, unknown> } | null)?.additional_kwargs;
    if (typeof kwargs?.reasoning_content === 'string') reasoning += kwargs.reasoning_content;

    const content = (chunk as { content?: unknown } | null)?.content;
    if (Array.isArray(content)) {
        for (const part of content) {
            if (typeof part !== 'object' || part === null) continue;
            const record = part as Record<string, unknown>;
            if (record.type === 'reasoning' && typeof record.text === 'string') reasoning += record.text;
        }
    }
    return reasoning;
};

/** 提取正文增量：兼容字符串与标准内容块数组两种形态 */
export const extractText = (chunk: unknown): string => {
    const content = (chunk as { content?: unknown } | null)?.content;
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    let text = '';
    for (const part of content) {
        if (typeof part !== 'object' || part === null) continue;
        const record = part as Record<string, unknown>;
        if (record.type === 'text' && typeof record.text === 'string') text += record.text;
    }
    return text;
};

/** 读取流式工具调用片段（参数是分片到达的 JSON 字符串，需要按 id 累加） */
export const readToolCallChunks = (chunk: unknown): ToolCallChunkInfo[] => {
    const raw = (chunk as { tool_call_chunks?: unknown } | null)?.tool_call_chunks;
    if (!Array.isArray(raw)) return [];
    const result: ToolCallChunkInfo[] = [];
    for (const item of raw) {
        if (typeof item !== 'object' || item === null) continue;
        const record = item as Record<string, unknown>;
        const id = typeof record.id === 'string' && record.id !== '' ? record.id : `index-${String(record.index ?? 0)}`;
        const info: ToolCallChunkInfo = {
            id,
            name: typeof record.name === 'string' ? record.name : '',
            args: typeof record.args === 'string' ? record.args : '',
        };
        // OpenAI 兼容流只有首个分片携带真实 id，后续参数分片仅有 index，透出供调用方归并
        if (typeof record.index === 'number') info.index = record.index;
        result.push(info);
    }
    return result;
};

/**
 * 流式引用还原器：把模型输出的引用锚点实时替换为可点击 markdown 链接，并在收尾时追加 References 列表。
 * 双标签 `[[auto index]][Source_N]` 与模型偷懒只写的裸锚点 `[Source_N]` 都会被还原。
 *
 * 由于锚点是跨 chunk 到达的，这里对「可能是锚点前缀」的尾部做有限长度回退，
 * 避免把半截锚点发到前端；超过锚点最大长度则判定为普通文本立即下发，防止流被卡住。
 */
export class CitationRewriter {
    private buffer = '';
    private readonly indexById = new Map<string, number>();
    /** 占位符是否已在流式过程中就地替换，避免收尾时重复追加 References */
    private referenceInlined = false;
    /** 就地替换时实际写入的 References 文本，用于收尾判断是否需要补齐 */
    private inlinedSection = '';
    /** 最近已下发文本的尾部：模型常自写 `## References` 标题，替换/补列表前据此避免标题重复 */
    private emittedTail = '';

    constructor(private readonly registry: SourceRegistry) {}

    /** 推入正文增量，返回可以安全下发的文本 */
    push(text: string): string {
        this.buffer += text;
        let output = '';

        while (true) {
            const matched = CITATION_PATTERN.exec(this.buffer);
            if (!matched || matched.index === undefined) break;
            output += this.buffer.slice(0, matched.index) + this.renderCitation(matched[1] ?? '');
            this.buffer = this.buffer.slice(matched.index + matched[0].length);
        }

        // 模型沿用 deepResearch 写作模板时会输出 References 占位符，就地替换为真实来源列表；
        // 标题若已被模型输出（`## References` + 占位符是提示词要求的标准形态），只插入列表避免标题重复
        const placeholderIndex = this.buffer.indexOf(REFERENCE_PLACEHOLDER);
        if (placeholderIndex >= 0) {
            const rawBefore = this.buffer.slice(0, placeholderIndex);
            const section = this.buildSectionFor(this.emittedTail + rawBefore, rawBefore).trim();
            this.buffer = this.buffer.replaceAll(REFERENCE_PLACEHOLDER, section);
            this.referenceInlined = true;
            this.inlinedSection = section;
        }

        const holdBack = this.holdBackLength(this.buffer);
        output += this.buffer.slice(0, this.buffer.length - holdBack);
        this.buffer = holdBack > 0 ? this.buffer.slice(this.buffer.length - holdBack) : '';
        if (output !== '') this.emittedTail = (this.emittedTail + output).slice(-EMITTED_TAIL_CHARS);
        return output;
    }

    /** 收尾：下发残余文本，并在末尾补齐 References 段落 */
    flush(): string {
        const pending = this.buffer;
        this.buffer = '';
        // 占位符已就地替换过：仅当替换时还没有可列出的来源，才在收尾补齐，避免 References 重复
        if (this.referenceInlined) {
            return this.inlinedSection === '' ? pending + this.buildSectionFor(this.emittedTail + pending) : pending;
        }
        const placeholderIndex = pending.indexOf(REFERENCE_PLACEHOLDER);
        if (placeholderIndex >= 0) {
            const rawBefore = pending.slice(0, placeholderIndex);
            const before = (this.emittedTail + rawBefore).replace(/\s+$/, '');
            return pending.replaceAll(REFERENCE_PLACEHOLDER, this.buildSectionFor(before, rawBefore).trim());
        }
        // 模型没写占位符：末尾补齐；若已自写 References 标题则只补列表，避免出现两个标题
        return pending + this.buildSectionFor(this.emittedTail + pending, pending);
    }

    private renderCitation(sourceId: string): string {
        const key = `[${sourceId}]`;
        let index = this.indexById.get(key);
        if (index === undefined) {
            index = this.indexById.size + 1;
            this.indexById.set(key, index);
        }
        return buildCitationLink(this.registry.get(key), index);
    }

    /**
     * 依据插入点之前的上下文构造 References 段落：
     * 标题已存在（模型自写）时只返回列表，否则带 `## References` 标题整体插入。
     * `rawTail` 是插入点前的原始文本，用于保证只补列表时与标题之间有换行分隔。
     */
    private buildSectionFor(context: string, rawTail = ''): string {
        if (this.indexById.size === 0) return '';
        const ordered = [...this.indexById.entries()]
            .map(([key, index]) => ({ key, index }))
            .sort((a, b) => a.index - b.index);
        const lines: string[] = [];
        for (const { key, index } of ordered) {
            const entry = this.registry.get(key);
            if (!entry) continue;
            lines.push(`${index}. [${entry.title}](${entry.url})`);
        }
        if (lines.length === 0) return '';
        if (endsWithReferencesHeading(context)) {
            return (/\n\s*$/.test(rawTail) ? '' : '\n\n') + `${lines.join('\n')}\n`;
        }
        return `\n\n## References\n\n${lines.join('\n')}\n`;
    }

    /**
     * 计算需要回退等待的尾部长度：
     * 只要尾部是引用锚点 / 占位符前缀的一部分（含单个 '['），就先不下发，等后续增量补齐。
     */
    private holdBackLength(value: string): number {
        // A. 缓冲区里还有未被消费的 [[auto index]]：后面的 [Source_N] 还没到齐，整体回退
        const markerIndex = value.lastIndexOf(CITATION_PREFIX);
        if (markerIndex >= 0) {
            const markerTail = value.length - markerIndex;
            if (markerTail <= MAX_CITATION_LENGTH) return markerTail;
        }

        // B. 出现未闭合的 [Source（裸锚点或双标签的后半段）：整体回退，等 ']' 到达
        const sourceIndex = value.lastIndexOf(SOURCE_PREFIX);
        if (sourceIndex >= 0 && !value.includes(']', sourceIndex)) {
            const sourceTail = value.length - sourceIndex;
            // 超过锚点最大长度即判定为普通文本立即下发，避免流被卡住
            if (sourceTail <= MAX_CITATION_LENGTH) return sourceTail;
        }

        // C. 否则只回退「可能是标记前缀」的最长尾部（'['、'[['、'[S'、'[R'…）
        let holdBack = 0;
        for (const token of HOLD_TOKENS) {
            const maxLength = Math.min(value.length, token.length - 1);
            for (let length = maxLength; length > 0; length -= 1) {
                if (token.startsWith(value.slice(value.length - length))) {
                    holdBack = Math.max(holdBack, length);
                    break;
                }
            }
        }
        return holdBack;
    }
}

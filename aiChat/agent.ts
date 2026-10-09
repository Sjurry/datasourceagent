import type { BaseMessage } from '@langchain/core/messages';
import { createAgent } from 'langchain';
import type { SourceRegistry } from '../../utilt/sourceRegistry.js';
import { getChatModel, type ChatModelOptions } from './model.js';
import { buildSystemPrompt } from './prompt.js';
import {
    CitationRewriter,
    extractReasoning,
    extractText,
    isAiMessage,
    isToolMessage,
    readToolCallChunks,
    type SseDelta,
} from './sseStream.js';
import { buildSearchTools } from './tools.js';
import { getSearchToolLabel } from '../deepResearch/searchTools.js';

export type ArtChatStreamParams = {
    /** 由服务端组装的 system prompt（DB 模板 + 分析开关），为空时不注入 */
    systemPrompt: string;
    /** 规范化后的对话消息（文章上下文 + 客户端多轮消息） */
    messages: BaseMessage[];
    /** 是否挂载联网检索工具；仅当用户在对话页选中「联网搜索」入口时为 true */
    enableTools: boolean;
    /** 本次请求的来源注册表，保证 [Source_N] 引用并发隔离 */
    registry: SourceRegistry;
    onDelta: (delta: SseDelta) => void;
    signal?: AbortSignal;
    modelOptions?: ChatModelOptions;
};

const parseToolQuery = (rawArgs: string): string => {
    if (rawArgs.trim() === '') return '';
    try {
        const parsed = JSON.parse(rawArgs) as Record<string, unknown>;
        const query = parsed.query;
        return typeof query === 'string' ? query.trim() : '';
    } catch {
        /* 参数仍在分片到达，等待补齐后再解析 */
        return '';
    }
};

/**
 * 以 LangChain createAgent 作为唯一编排入口执行一次文章分析对话：
 * 未启用工具时 tools 为空数组，Agent 退化为「system + 多轮消息 → 模型」的等价链路。
 */
export async function runArtChatStream(params: ArtChatStreamParams): Promise<void> {
    const { systemPrompt, messages, enableTools, registry, onDelta, signal, modelOptions } = params;

    const model = getChatModel(modelOptions ?? {});
    const tools = enableTools ? buildSearchTools(registry) : [];
    const rewriter = new CitationRewriter(registry);

    const resolvedSystemPrompt = buildSystemPrompt(systemPrompt, enableTools);
    const agent = createAgent({
        model,
        tools,
        ...(resolvedSystemPrompt !== '' ? { systemPrompt: resolvedSystemPrompt } : {}),
    });

    /** 工具调用参数是分片到达的，按 id 累加后解析出检索词 */
    const pendingCalls = new Map<string, { name: string; args: string }>();
    const announcedCalls = new Set<string>();
    /** OpenAI 兼容流只有首个分片携带真实 id/name，后续参数分片仅有 index，用此映射把参数归并回真实调用 */
    const callIndexIds = new Map<number, string>();

    const announceToolCalls = (chunk: unknown) => {
        for (const call of readToolCallChunks(chunk)) {
            if (typeof call.index === 'number' && !call.id.startsWith('index-')) {
                callIndexIds.set(call.index, call.id);
            }
            const callId =
                typeof call.index === 'number' ? (callIndexIds.get(call.index) ?? call.id) : call.id;
            const previous = pendingCalls.get(callId);
            const merged = {
                name: call.name !== '' ? call.name : (previous?.name ?? ''),
                args: `${previous?.args ?? ''}${call.args}`,
            };
            pendingCalls.set(callId, merged);

            if (announcedCalls.has(callId)) continue;
            const query = parseToolQuery(merged.args);
            if (query === '') continue;
            announcedCalls.add(call.id);
            // 标注本次使用的检索工具（如 Serper / Tavily），未知工具名时退化为原始文案
            const label = getSearchToolLabel(merged.name);
            onDelta({ reasoning: `\n正在联网检索${label !== '' ? `（${label}）` : ''}：${query}\n` });
        }
    };

    const streamConfig = { streamMode: ['messages', 'updates'] };
    const stream = await agent.stream(
        { messages },
        (signal === undefined ? streamConfig : { ...streamConfig, signal }) as unknown as Parameters<
            typeof agent.stream
        >[1]
    );

    for await (const raw of stream as unknown as AsyncIterable<[string, unknown]>) {
        const [mode, payload] = raw;
        if (mode !== 'messages') continue;

        const message = Array.isArray(payload) ? payload[0] : payload;

        if (isToolMessage(message)) {
            onDelta({ reasoning: '\n检索结果已返回，正在整理资料…\n' });
            continue;
        }
        if (!isAiMessage(message)) continue;

        const reasoning = extractReasoning(message);
        if (reasoning !== '') onDelta({ reasoning });

        announceToolCalls(message);

        const text = extractText(message);
        if (text === '') continue;
        const safeText = rewriter.push(text);
        if (safeText !== '') onDelta({ content: safeText });
    }

    const tail = rewriter.flush();
    if (tail !== '') onDelta({ content: tail });
}

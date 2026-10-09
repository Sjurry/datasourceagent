import { AIMessage, HumanMessage, SystemMessage, trimMessages, type BaseMessage } from '@langchain/core/messages';
import { logger } from '../../utilt/logger.js';

export type ChatRole = 'system' | 'user' | 'assistant';

/** 客户端传入的消息结构（与旧转发实现的入参保持一致） */
export type ChatInputMessage = { role: ChatRole; content: string };

/** 默认历史预算：按偏保守的估算约为 3.6 万字符，留出足够的输出空间 */
const DEFAULT_MAX_HISTORY_TOKENS = 24_000;

export const normalizeRole = (role: unknown): ChatRole => {
    if (role === 'system') return 'system';
    if (role === 'assistant') return 'assistant';
    return 'user';
};

/** 清洗客户端消息：角色白名单 + 过滤空内容 */
export const normalizeMessages = (raw: unknown): ChatInputMessage[] => {
    if (!Array.isArray(raw)) return [];
    const messages: ChatInputMessage[] = [];
    for (const item of raw) {
        if (typeof item !== 'object' || item === null) continue;
        const record = item as Record<string, unknown>;
        const content = typeof record.content === 'string' ? record.content : '';
        if (!content.trim()) continue;
        messages.push({ role: normalizeRole(record.role), content });
    }
    return messages;
};

export const toBaseMessages = (messages: ChatInputMessage[]): BaseMessage[] =>
    messages.map((message) => {
        if (message.role === 'system') return new SystemMessage(message.content);
        if (message.role === 'assistant') return new AIMessage(message.content);
        return new HumanMessage(message.content);
    });

/**
 * 粗略 token 估算：中英文混排下按 1.5 字符 ≈ 1 token 偏保守估算，
 * 宁可提前裁剪也不要把请求顶到上下文窗口边缘。
 */
export const estimateTokens = (messages: BaseMessage[]): number => {
    let chars = 0;
    for (const message of messages) {
        if (typeof message.content === 'string') chars += message.content.length;
        else if (Array.isArray(message.content)) {
            for (const part of message.content) {
                if (typeof part === 'object' && part !== null && 'text' in part && typeof part.text === 'string') {
                    chars += part.text.length;
                }
            }
        }
    }
    return Math.ceil(chars / 1.5);
};

const resolveMaxHistoryTokens = (): number => {
    const configured = Number(process.env.AI_CHAT_MAX_HISTORY_TOKENS?.trim());
    return Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_MAX_HISTORY_TOKENS;
};

/**
 * 按 token 预算裁剪历史：保留首条 SystemMessage，从最近的消息往前保留，
 * 并保证裁剪后仍以 user 消息开头和结尾，避免出现孤立的 assistant 消息。
 *
 * 会话历史由前端持久化（服务端无 checkpointer），因此这里用确定性的窗口裁剪而非摘要压缩，
 * 避免每次请求都触发一次额外的摘要模型调用。
 */
export const trimHistory = async (messages: BaseMessage[]): Promise<BaseMessage[]> => {
    if (messages.length <= 1) return messages;
    try {
        const trimmed = await trimMessages(messages, {
            maxTokens: resolveMaxHistoryTokens(),
            strategy: 'last',
            tokenCounter: estimateTokens,
            includeSystem: true,
            startOn: 'human',
            endOn: 'human',
            allowPartial: false,
        });
        return trimmed.length > 0 ? trimmed : messages;
    } catch (error) {
        logger.error('[aiChat] 历史裁剪失败，回退为原始消息：', error);
        return messages;
    }
};

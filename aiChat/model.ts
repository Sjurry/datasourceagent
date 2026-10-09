import { ChatOpenAI } from '@langchain/openai';
import { asserts } from '../asserts.js';

/** 单次请求可覆盖的模型参数，未传时回落到环境变量 */
export type ChatModelOptions = {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
};

/** 默认超时：5 分钟 */
const DEFAULT_TIMEOUT_MS = 300_000;
const DEFAULT_MODEL = 'gpt-4o';

/** 按「模型名 + 参数」缓存实例：客户端内部持有连接池，逐请求重建会带来额外握手开销 */
const modelCache = new Map<string, ChatOpenAI>();
/** 缓存上限：模型参数组合有限，超过则整体清空，避免长期运行后无界增长 */
const MAX_CACHED_MODELS = 64;

/** 只接受有限且为正的数值参数，避免把 NaN / 负数透传给 SDK */
const sanitizePositive = (value: number | undefined): number | undefined =>
    value !== undefined && Number.isFinite(value) && value > 0 ? value : undefined;

/** 只接受有限数值（temperature 允许 0），否则回退默认值 */
const sanitizeFinite = (value: number | undefined): number | undefined =>
    value !== undefined && Number.isFinite(value) ? value : undefined;

const resolveOpenAiEnv = () => {
    const apiKey = process.env.OPENAI_API_KEY?.trim();
    const baseURL = process.env.OPENAI_BASE_URL?.trim();
    asserts(apiKey !== undefined && apiKey !== '', 'OPENAI_API_KEY 未配置');
    asserts(baseURL !== undefined && baseURL !== '', 'OPENAI_BASE_URL 未配置');
    const timeout = Number(process.env.OPENAI_TIMEOUT_MS?.trim()) || DEFAULT_TIMEOUT_MS;
    return { apiKey, baseURL, timeout };
};

/**
 * 构造走自建 OpenAI 兼容网关的 ChatOpenAI。
 *
 * 注意：项目使用自定义 baseURL，必须显式传 apiKey + configuration.baseURL，
 * 不能依赖 SDK 默认的 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量名。
 */
export function getChatModel(options: ChatModelOptions = {}): ChatOpenAI {
    const { apiKey, baseURL, timeout } = resolveOpenAiEnv();
    const model = options.model?.trim() || process.env.OPENAI_MODEL?.trim() || DEFAULT_MODEL;
    const temperature = sanitizeFinite(options.temperature) ?? 0;
    const maxTokens = sanitizePositive(options.maxTokens);
    const topP = sanitizePositive(options.topP);
    const cacheKey = JSON.stringify({
        baseURL,
        model,
        temperature,
        maxTokens: maxTokens ?? null,
        topP: topP ?? null,
        timeout,
    });

    const cached = modelCache.get(cacheKey);
    if (cached) return cached;

    if (modelCache.size >= MAX_CACHED_MODELS) modelCache.clear();

    const instance = new ChatOpenAI({
        apiKey,
        model,
        temperature,
        timeout,
        configuration: { baseURL },
        ...(maxTokens !== undefined ? { maxTokens } : {}),
        ...(topP !== undefined ? { topP } : {}),
    });
    modelCache.set(cacheKey, instance);
    return instance;
}

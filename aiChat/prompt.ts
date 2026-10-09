import { logger } from "../../utilt/logger.js";

/** 联网检索模式下附加的引用规范，复用 deepResearch 已有的 [Source_N] 锚点约定 */
export const TOOL_CITATION_PROMPT = `【联网检索要求】
- 可用检索工具：tavily_search（通用网页）、serper_search（谷歌实时资讯）、exa_search（语义检索，适合概念性、技术性与中长文本内容）、arxiv_search（学术预印本）、wikipedia_search（背景词条）。请根据问题类型自主选择；需要时组合多个工具交叉核验。
- 必须优先检索当前年份的消息，过滤无效或者旧消息
- 优先基于检索结果回答；检索不到有效信息时如实说明，不要编造。
- 引用检索结果时只使用工具返回的 [Source_N] 锚点，按 [[auto index]][Source_N] 格式标注；同一处引用多个来源时分开放。
- 不要输出真实 URL，也不要使用工具结果中未出现的 Source 编号。`;

const DEFAULT_TIME_ZONE = 'Asia/Shanghai';

/**
 * 日期格式化器按「天」精度缓存在模块级：
 * 注入到日即可满足「今天 / 最新 / 近期 / 今年」的表述，
 * 同时让 system prompt 一天才变化一次，避免每分钟变化影响网关的前缀缓存。
 */
const dateFormatters = new Map<string, Intl.DateTimeFormat>();

const getDateFormatter = (timeZone: string): Intl.DateTimeFormat => {
    const cached = dateFormatters.get(timeZone);
    if (cached) return cached;
    const formatter = new Intl.DateTimeFormat('zh-CN', {
        timeZone,
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long',
    });
    dateFormatters.set(timeZone, formatter);
    return formatter;
};

/** 时区：默认北京时间，可用环境变量 AI_CHAT_TIME_ZONE 覆盖 */
const resolveTimeZone = (): string => process.env.AI_CHAT_TIME_ZONE?.trim() || DEFAULT_TIME_ZONE;

/**
 * 运行时注入当前日期。
 *
 * 模型的知识截止日期远早于当前日期，不注入会让它把「今天 / 最新 / 近期」理解成训练数据里的日期，
 * 从而检索出过期信息（例如把 2026 年当成 2025 年）。
 *
 * `now` / `timeZone` 可显式传入，便于测试与调用方固定行为。
 */
export const buildCurrentTimePrompt = (
    now: Date = new Date(),
    timeZone: string = resolveTimeZone()
): string =>
    `【当前时间】${getDateFormatter(timeZone).format(now)}（时区 ${timeZone}）。涉及"今天""最新""近期""今年"等时间表述时，一律以此日期为准，不要使用训练数据中的日期。所有的搜索，必须以【当前时间】为准，搜索最新消息，过滤无效或者旧消息。`;

/** 组装 system prompt：DB 模板 + 当前日期 +（可选）联网引用规范 */
export const buildSystemPrompt = (
    systemPrompt: string,
    enableTools: boolean,
    now: Date = new Date(),
    timeZone: string = resolveTimeZone()
): string => {
    const segments = [systemPrompt.trim(), buildCurrentTimePrompt(now, timeZone)];
    if (enableTools) segments.push(TOOL_CITATION_PROMPT);
    return segments.filter((segment) => segment !== '').join('\n\n');
};

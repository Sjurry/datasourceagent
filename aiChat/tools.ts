import { tool } from 'langchain';
import { z } from 'zod';
import { executeSearchToolWith } from '../deepResearch/searchTools.js';
import type { SourceRegistry } from '../../utilt/sourceRegistry.js';

/** 检索工具集选项 */
export type SearchToolsOptions = {
    /**
     * 是否挂载 wikipedia_search，默认 true。
     * 深度调研的 PLANNING_PROMPT / RESEARCH_PROMPT 只声明了 tavily_search / serper_search / exa_search / arxiv_search，
     * 因此 deep 链路显式传 false，避免给模型提供提示词未覆盖的工具。
     */
    includeWikipedia?: boolean;
};

/**
 * 把 deepResearch 已有的 Tavily / Serper / Exa / arXiv / 维基百科检索能力包装成 LangChain 工具。
 *
 * 每个请求传入独立的 SourceRegistry，保证 [Source_N] 引用锚点不会在并发请求间串号；
 * 工具返回的 JSON 中 url 字段仍是锚点，最终由 CitationRewriter 还原为可点击链接。
 */
export const buildSearchTools = (registry: SourceRegistry, options: SearchToolsOptions = {}) => {
    const all = [
        tool(
            async (input) =>
                executeSearchToolWith(registry, 'tavily_search', {
                    query: input.query,
                    max_results: input.max_results ?? 5,
                }),
            {
                name: 'tavily_search',
                description:
                    '通用网页搜索，返回与查询最相关的网页片段与 [Source_N] 引用锚点，适合新闻、行业报告与一般性资料。',
                schema: z.object({
                    query: z.string().describe('搜索关键词'),
                    max_results: z.number().int().min(1).max(20).optional().describe('返回结果条数，默认 5'),
                }),
            }
        ),
        tool(
            async (input) =>
                executeSearchToolWith(registry, 'serper_search', {
                    query: input.query,
                    max_results: input.max_results ?? 5,
                }),
            {
                name: 'serper_search',
                description:
                    'Serper.dev 谷歌网页搜索，返回与查询最相关的网页片段与 [Source_N] 引用锚点，适合时效性强的最新资讯与需要谷歌结果的场景。',
                schema: z.object({
                    query: z.string().describe('搜索关键词'),
                    max_results: z.number().int().min(1).max(20).optional().describe('返回结果条数，默认 5'),
                }),
            }
        ),
        tool(
            async (input) =>
                executeSearchToolWith(registry, 'exa_search', {
                    query: input.query,
                    max_results: input.max_results ?? 5,
                }),
            {
                name: 'exa_search',
                description:
                    'Exa 语义检索，基于向量语义匹配网页与文章内容，适合概念性、技术性、中长文本与深度内容的检索；与关键词型搜索（tavily_search / serper_search）交叉使用可提升覆盖度。',
                schema: z.object({
                    query: z.string().describe('搜索关键词或自然语言描述'),
                    max_results: z.number().int().min(1).max(20).optional().describe('返回结果条数，默认 5'),
                }),
            }
        ),
        tool(
            async (input) =>
                executeSearchToolWith(registry, 'arxiv_search', {
                    query: input.query,
                    max_results: input.max_results ?? 5,
                }),
            {
                name: 'arxiv_search',
                description: '检索 arXiv 预印本论文，用于学术性的研究资料与关键概念核验。',
                schema: z.object({
                    query: z.string().describe('论文检索关键词'),
                    max_results: z.number().int().min(1).max(20).optional().describe('返回结果条数，默认 5'),
                }),
            }
        ),
        tool(
            async (input) =>
                executeSearchToolWith(registry, 'wikipedia_search', {
                    query: input.query,
                    sentences: input.sentences ?? 3,
                }),
            {
                name: 'wikipedia_search',
                description: '查询英文维基百科词条的摘要，适合快速核对背景、定义与人物/机构信息。',
                schema: z.object({
                    query: z.string().describe('查询关键词'),
                    sentences: z.number().int().min(1).max(10).optional().describe('摘要句数，默认 3'),
                }),
            }
        ),
    ];

    return options.includeWikipedia === false
        ? all.filter((item) => item.name !== 'wikipedia_search')
        : all;
};

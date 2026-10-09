import { randomUUID } from 'node:crypto';
import type { ResultSetHeader, RowDataPacket } from 'mysql2';
import { pool } from '../utilt/db.js';
import { runAuthorityTraceAgent, type TraceArticleInput, type TraceCandidateOutput } from './authorityTrace/agent.js';

export type TraceStatus = 'queued' | 'running' | 'completed' | 'failed';
export type TraceSourceType = 'government' | 'official' | 'institution' | 'authoritative_media' | 'media' | 'unknown';
export type AuthorityTraceCandidate = TraceCandidateOutput;

export interface AuthorityTraceTask {
    taskId: string;
    newsId: number;
    status: TraceStatus;
    intent: string;
    claims: string[];
    candidates: AuthorityTraceCandidate[];
    error?: string;
    createdAt: string;
}

interface NewsRow extends RowDataPacket, TraceArticleInput {
    id: number;
    url: string | null;
    media: string | null;
    publish_time: string | null;
    l1: string | null;
    l2_config: string | null;
}

const tasks = new Map<string, AuthorityTraceTask>();
const clean = (value: unknown) => String(value ?? '').replace(/\s+/g, ' ').trim();

export const createTraceTask = async (newsId: number): Promise<AuthorityTraceTask> => {
    const [rows] = await pool.query<NewsRow[]>(`SELECT id, title, title_cn, content, content_cn, summary, summary_cn, url, media, publish_time, l1, l2_config FROM news WHERE id = ? AND del_stat = 0 LIMIT 1`, [newsId]);
    const article = rows[0];
    if (!article) throw new Error('文章不存在');
    const task: AuthorityTraceTask = {
        taskId: `trace_${Date.now()}_${randomUUID().slice(0, 8)}`,
        newsId,
        status: 'queued',
        intent: 'unknown',
        claims: [],
        candidates: [],
        createdAt: new Date().toISOString(),
    };
    tasks.set(task.taskId, task);
    void runAuthorityTrace(task, article);
    return task;
};

const runAuthorityTrace = async (task: AuthorityTraceTask, article: NewsRow) => {
    task.status = 'running';
    try {
        const result = await runAuthorityTraceAgent(article, { taskId: task.taskId, newsId: task.newsId });
        task.intent = result.intent;
        task.claims = result.claims;
        task.candidates = result.candidates;
        task.status = 'completed';
    } catch (error) {
        task.status = 'failed';
        task.error = error instanceof Error ? error.message : String(error);
    }
};

export const getTraceTask = (taskId: string) => tasks.get(taskId);

export const confirmTraceCandidate = async (taskId: string, candidateId: string, category?: { l1?: string; l2_config?: string; keyword?: string }) => {
    const task = tasks.get(taskId);
    if (!task) throw new Error('溯源任务不存在');
    const candidate = task.candidates.find((item) => item.id === candidateId);
    if (!candidate) throw new Error('溯源候选不存在');
    if (candidate.supportLevel === 'unknown') throw new Error('候选来源缺少可用证据');

    const [sourceRows] = await pool.query<NewsRow[]>(`SELECT l1, l2_config, publish_time FROM news WHERE id = ? LIMIT 1`, [task.newsId]);
    const sourceArticle = sourceRows[0];
    const values = [
        candidate.url,
        candidate.title,
        candidate.content,
        sourceArticle?.publish_time ?? new Date(),
        candidate.publisher,
        candidate.sourceType,
        'authority_trace',
        clean(category?.l1) || clean(sourceArticle?.l1),
        clean(category?.l2_config) || clean(sourceArticle?.l2_config),
        clean(category?.keyword) || candidate.title.slice(0, 120),
    ];
    const [result] = await pool.query<ResultSetHeader>(`INSERT IGNORE INTO news (url,title,content,publish_time,media,data_source_type,tag,l1,l2_config,keyword) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, values);
    const [rows] = await pool.query<NewsRow[]>(`SELECT id, title, title_cn, content, content_cn, summary, summary_cn, url, media, publish_time, l1, l2_config FROM news WHERE url = ? LIMIT 1`, [candidate.url]);
    const insertedNewsId = Number(rows[0]?.id ?? result.insertId ?? 0);
    if (!insertedNewsId) throw new Error('新闻入库成功后未能获取新闻 ID');
    return {
        newsId: insertedNewsId,
        inserted: Number(result.affectedRows ?? 0) > 0,
        pipeline: { status: 'simulated', jobId: `sim_${Date.now()}_${randomUUID().slice(0, 8)}` },
        article: rows[0],
    };
};

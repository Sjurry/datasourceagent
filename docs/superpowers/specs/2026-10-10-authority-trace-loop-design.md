# Authority Trace 5轮Loop设计（2026-10-10）

## 1. 目标
时效新闻溯源：最多5轮，每轮必全网+指定域名双路，非权威丢弃，每轮换角度新词，可提前退出。

## 2. 架构
规则门 -> 试搜门 -> loop×5[plan -> 双路搜 -> 非权威丢弃 -> rank] -> 输出

## 3. 入口双门（选项C）
- 规则门：无实质事实/纯观点/正文<50字直接退出，不检索。
- 试搜门：1-2个种子query双路试搜，有direct+authority>=4直接返回，否则进loop。

## 4. Loop（最多5轮）
- 轮前退出条件（任一）：sufficient / loop>=5 / 连续两轮无新URL。
- plan策略A（换角度优先，域名不变）：保留核心实体，换同义/时间/英文/上下游之一，与历史query去重。
- 搜索：每轮必全网+指定域名双路并发。
- 过滤：authorityLevel<4直接丢弃，不进rank，不调LLM裁判。
- rank：LLM裁判打分排序，更新sufficient与无新URL计数。

## 5. 数据流
State新增：historyQueries、seenUrls、noNewUrlStreak。plan写historyQueries，search写seenUrls，rank更新sufficient。

## 6. 错误处理
Transient自动重试3次；可恢复错误进error_handler全网兜底；未知错误上抛。

## 7. 测试
单轮命中直接退；5轮上限；非权威过滤；连续无新URL退出；纯观点规则门退出。

# Agent graph

Generated from `build_graph` (`graph.get_graph().draw_mermaid()`).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	intent(intent)
	rewrite(rewrite)
	search(search)
	select_tables(select_tables)
	load_columns(load_columns)
	generate_sql(generate_sql)
	execute_sql(execute_sql)
	answer(answer)
	clarify(clarify)
	meta(meta)
	off_topic(off_topic)
	__end__([<p>__end__</p>]):::last
	__start__ --> intent;
	execute_sql --> answer;
	generate_sql --> execute_sql;
	intent -.-> clarify;
	intent -.-> meta;
	intent -.-> off_topic;
	intent -.-> rewrite;
	load_columns --> generate_sql;
	rewrite --> search;
	search --> select_tables;
	select_tables --> load_columns;
	answer --> __end__;
	clarify --> __end__;
	meta --> __end__;
	off_topic --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

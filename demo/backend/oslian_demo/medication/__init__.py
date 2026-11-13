"""Medication Safety: parse a case, rank candidates symbolically, explain them.

    engine/          the five-layer pharmacology reasoning engine
    data/            hand-authored drug and disease knowledge (research estimates)
    case_parser.py   free text -> structured fields (rule-based, LLM-assisted)
    case_targets.py  indication -> treatment target states
    disease_world.py local disease "wiki" read from the organ JSONs
    agent.py         the deterministic step pipeline the UI streams
    orchestrator.py  the alternative LLM tool-calling pipeline
    llm_client.py    the only place that talks to an LLM provider
    routes.py        Flask blueprint
"""

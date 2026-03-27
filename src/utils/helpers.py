def table_exists(session, fully_qualified_name: str) -> bool:
    try:
        _ = session.table(fully_qualified_name).schema
        return True
    except Exception:
        return False

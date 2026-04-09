from snowflake.ml.registry import Registry


def table_exists(session, fully_qualified_name: str) -> bool:
    try:
        _ = session.table(fully_qualified_name).schema
        return True
    except Exception:
        return False


def get_or_create_registry(session, database: str, schema: str) -> Registry:
    current_schema = session.get_current_schema()
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()
    mr = Registry(session=session, database_name=database, schema_name=schema)
    session.sql(f"USE SCHEMA {current_schema}").collect()
    return mr

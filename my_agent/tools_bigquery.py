"""Real BigQuery tools using google-cloud-bigquery."""

import os

from google.cloud import bigquery


PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "plurigrid")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT)
    return _client


def bigquery_query(sql: str, project: str = "plurigrid", max_rows: int = 100) -> dict:
    """Execute a BigQuery SQL query and return results.

    Args:
        sql: The SQL query to execute.
        project: GCP project ID for billing.
        max_rows: Maximum number of rows to return.
    """
    try:
        client = _get_client()
        job = client.query(sql, project=project)
        rows = []
        for row in job.result():
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
        return {
            "sql": sql,
            "rows": rows,
            "total_rows": job.result().total_rows,
            "bytes_processed": job.total_bytes_processed,
            "status": "value",
        }
    except Exception as e:
        return {"sql": sql, "error": str(e), "status": "contradiction"}


def bigquery_table_info(dataset: str, table: str) -> dict:
    """Get schema information for a BigQuery table.

    Args:
        dataset: BigQuery dataset ID.
        table: Table name.
    """
    try:
        client = _get_client()
        ref = client.dataset(dataset).table(table)
        tbl = client.get_table(ref)
        return {
            "dataset": dataset,
            "table": table,
            "schema": [
                {"name": f.name, "type": f.field_type, "mode": f.mode}
                for f in tbl.schema
            ],
            "row_count": tbl.num_rows,
            "size_bytes": tbl.num_bytes,
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def bigquery_list_datasets() -> dict:
    """List all datasets in the project."""
    try:
        client = _get_client()
        datasets = list(client.list_datasets())
        return {
            "datasets": [ds.dataset_id for ds in datasets],
            "count": len(datasets),
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}


def bigquery_list_tables(dataset: str) -> dict:
    """List all tables in a BigQuery dataset.

    Args:
        dataset: BigQuery dataset ID.
    """
    try:
        client = _get_client()
        tables = list(client.list_tables(dataset))
        return {
            "dataset": dataset,
            "tables": [t.table_id for t in tables],
            "count": len(tables),
            "status": "value",
        }
    except Exception as e:
        return {"error": str(e), "status": "contradiction"}

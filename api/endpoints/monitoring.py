"""
F1PA API - Monitoring Endpoints

Endpoints to access ML monitoring reports (Evidently).
"""
import sys
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.auth import get_current_user

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

REPORTS_DIR = PROJECT_ROOT / "monitoring" / "evidently" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/drift/reports")
async def list_drift_reports(username: str = Depends(get_current_user)) -> List[str]:
    """
    List all available drift reports.

    Returns:
        List of report file names
    """
    reports = sorted(REPORTS_DIR.glob("*.html"), reverse=True)
    return [r.name for r in reports]


@router.get("/drift/reports/{report_name}")
async def get_drift_report(
    report_name: str,
    username: str = Depends(get_current_user)
):
    """
    Retrieve a specific drift report.

    Args:
        report_name: Report file name (with .html extension)

    Returns:
        HTML content of the report
    """
    # Security: validate that name does not contain path traversal
    if ".." in report_name or "/" in report_name or "\\" in report_name:
        raise HTTPException(status_code=400, detail="Invalid report name")

    report_path = REPORTS_DIR / report_name

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {report_name} not found")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=report_name
    )


@router.get("/drift/latest")
async def get_latest_drift_report(username: str = Depends(get_current_user)):
    """
    Retrieve the latest generated drift report.

    Returns:
        HTML content of the latest report
    """
    reports = sorted(REPORTS_DIR.glob("*.html"), reverse=True)

    if not reports:
        raise HTTPException(
            status_code=404,
            detail="No drift reports available. Generate one first."
        )

    latest_report = reports[0]

    return FileResponse(
        path=str(latest_report),
        media_type="text/html",
        filename=latest_report.name
    )


@router.get("/status")
async def get_monitoring_status(username: str = Depends(get_current_user)):
    """
    Retrieve monitoring status.

    Returns:
        Information about available reports
    """
    reports = list(REPORTS_DIR.glob("*.html"))

    return {
        "total_reports": len(reports),
        "latest_report": reports[-1].name if reports else None,
        "reports_directory": str(REPORTS_DIR)
    }

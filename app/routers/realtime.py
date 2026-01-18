from fastapi import APIRouter, Depends, HTTPException, Query

from src.core.realtime_manager import RealtimeManager, RealtimeConfig
from app.dependencies import get_realtime_manager
from app.schemas import RealtimeConfigRequest, RealtimeStatusResponse


router = APIRouter(prefix="/api/realtime", tags=["Realtime"])

@router.post("/start", response_model=RealtimeStatusResponse)
async def start_realtime_session(
    config_req: RealtimeConfigRequest,
    realtime_manager: RealtimeManager = Depends(get_realtime_manager)
):
    """Start a new realtime analysis session"""
    config = RealtimeConfig(
        camera_id=config_req.camera_id,
        display_width=config_req.display_width,
        min_det_score=config_req.min_det_score,
        save_json=config_req.save_json,
        save_video=config_req.save_video,
        visualize=config_req.visualize
    )
    
    try:
        session = realtime_manager.start_session(config)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return RealtimeStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        start_time=session.start_time.isoformat(),
        config=session.config.dict(),
        output_dir=session.output_dir,
        error=session.error
    )

@router.post("/stop")
async def stop_realtime_session(
    realtime_manager: RealtimeManager = Depends(get_realtime_manager)
):
    """Stop the current realtime analysis session"""
    realtime_manager.stop_session()
    return {"success": True, "message": "Realtime session stopped"}

@router.get("/status", response_model=RealtimeStatusResponse)
async def get_realtime_status(
    realtime_manager: RealtimeManager = Depends(get_realtime_manager)
):
    """Get current realtime session status"""
    session = realtime_manager.get_status()
    
    return RealtimeStatusResponse(
        session_id=session.session_id,
        status=session.status.value,
        start_time=session.start_time.isoformat(),
        config=session.config.dict(),
        output_dir=session.output_dir,
        error=session.error
    )

@router.get("/logs")
async def get_realtime_logs(
    limit: int = Query(100, ge=1, le=500),
    realtime_manager: RealtimeManager = Depends(get_realtime_manager)
):
    """Get realtime session logs"""
    logs = realtime_manager.get_logs(limit=limit)
    return {"logs": logs, "total": len(logs)}

"""Isolated NVIDIA OVRT implementation of the Newton viewer contract."""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from blacknode.pkg.blacknode_newton.viewer_contract import register_viewer


def _workspace_scene_path(path: Any) -> str:
    """Map private OVRT wrapper prims back to public workspace scene paths."""
    value = str(path or "")
    if value == "/BlacknodeOVRT/Ground" or value.startswith("/BlacknodeOVRT/Ground/"):
        return "/Blacknode/Ground"
    return value


_VIEWER_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blacknode Newton · OVRT</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#080b11;color:#edf3ff}*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#080b11}#v{position:fixed;inset:0;display:grid;place-items:center;cursor:default;user-select:none;touch-action:none}#v.drag{cursor:grabbing}#s{width:100%;height:100%;object-fit:contain;pointer-events:none;-webkit-user-drag:none}#e{position:absolute;inset:0;display:grid;place-items:center;background:radial-gradient(circle at 50% 42%,#172131,#080b11 65%)}#c{max-width:520px;padding:24px;text-align:center}.spin{width:34px;height:34px;margin:0 auto 15px;border:3px solid #273449;border-top-color:#76b900;border-radius:50%;animation:r 1s linear infinite}@keyframes r{to{transform:rotate(360deg)}}#s1{font-weight:700;font-size:15px}#d{margin-top:8px;color:#9ba9bd;font-size:13px;line-height:1.45}#h{position:fixed;left:12px;top:12px;padding:8px 10px;border:1px solid #ffffff18;border-radius:8px;background:#080b11bb;font-size:11px;line-height:1.5;pointer-events:none}#h strong{color:#76b900}#help{position:fixed;right:12px;bottom:12px;padding:7px 9px;border-radius:7px;background:#080b11aa;color:#aab5c5;font-size:10px;pointer-events:none}
#m{position:fixed;left:50%;top:12px;z-index:4;display:flex;gap:3px;padding:4px;transform:translateX(-50%);border:1px solid #ffffff20;border-radius:9px;background:#080b11dd;backdrop-filter:blur(10px)}#m button{padding:5px 8px;border:1px solid transparent;border-radius:6px;background:transparent;color:#aeb9c9;cursor:pointer;font:600 10px Inter,system-ui,sans-serif}#m button:hover{color:#fff;background:#ffffff10}#m button.on{border-color:#76b900aa;background:#76b90022;color:#dfffad}
#tools{position:fixed;left:12px;top:50%;z-index:8;display:grid;gap:3px;padding:4px;transform:translateY(-50%);border:1px solid #ffffff20;border-radius:9px;background:#080b11dd;box-shadow:0 8px 24px #0006;backdrop-filter:blur(10px)}#tools button{position:relative;display:grid;width:34px;height:34px;padding:0;border:1px solid transparent;border-radius:6px;background:transparent;color:#aeb9c9;place-items:center;cursor:pointer}#tools button:hover{color:#fff;background:#ffffff10}#tools button.on{border-color:#76b900aa;background:#76b90022;color:#dfffad;box-shadow:inset 3px 0 #76b900}#tools svg{width:19px;height:19px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}#tools kbd{position:absolute;right:2px;bottom:1px;color:#78869a;font:700 8px Inter,system-ui,sans-serif}#tools button.on kbd{color:#b9e47c}
#g{position:fixed;z-index:7;display:none;width:0;height:0;pointer-events:none;opacity:0}#g[data-tool="select"]{display:none!important}#g .move-handles,#g .scale-handles,#g .rotate-handles{display:none}#g[data-tool="move"] .move-handles,#g[data-tool="scale"] .scale-handles,#g[data-tool="rotate"] .rotate-handles{display:block}#g.is-picking .move-handles,#g.is-picking .scale-handles,#g.is-picking .rotate-handles{display:none}
#g .gizmo-axis{--axis:#fff;position:absolute;left:0;top:-5px;width:58px;height:10px;padding:0;border:0;background:linear-gradient(transparent 3px,var(--axis) 3px,var(--axis) 7px,transparent 7px);transform-origin:0 50%;pointer-events:auto;cursor:grab}#g .move-axis::after{content:"";position:absolute;right:-1px;top:0;border-left:10px solid var(--axis);border-top:5px solid transparent;border-bottom:5px solid transparent}#g .scale-axis::after{content:"";position:absolute;right:-3px;top:0;width:10px;height:10px;background:var(--axis);box-shadow:0 0 0 1px #0008}#g .gizmo-axis span{position:absolute;right:-18px;top:-4px;color:var(--axis);font:800 11px Inter,system-ui,sans-serif;text-shadow:0 1px 2px #000}#g .gizmo-handle:active{cursor:grabbing;filter:brightness(1.5)}#g .axis-x{--axis:#ff4d45}#g .axis-y{--axis:#66d15c}#g .axis-z{--axis:#4795ff}
#g .rotate-handles{position:absolute;left:-52px;top:-52px;width:104px;height:104px;overflow:visible;pointer-events:none}#g .rotate-ring{fill:none;stroke:var(--axis);stroke-width:4;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke;pointer-events:stroke;cursor:grab;filter:drop-shadow(0 0 1px #000)}#g .scale-uniform{position:absolute;left:-6px;top:-6px;width:12px;height:12px;padding:0;border:1px solid #fff;background:#ffb21a;pointer-events:auto;cursor:grab;transform:rotate(45deg)}
</style></head><body><div id="v"><img id="s" src="/stream.mjpg" draggable="false"><div id="e"><div id="c"><div class="spin"></div><div id="s1">Starting NVIDIA OVRT</div><div id="d">The first render initializes and caches RTX shaders.</div></div></div></div><div id="tools" role="toolbar" aria-label="Viewport tools"><button data-tool="select" class="on" title="Select (Q)" aria-label="Select tool (Q)"><svg viewBox="0 0 24 24"><path d="M5 3l12 9-6 1.5L8.5 19z"/></svg><kbd>Q</kbd></button><button data-tool="move" title="Move (W)" aria-label="Move tool (W)"><svg viewBox="0 0 24 24"><path d="M12 2v20M2 12h20M12 2l-3 3m3-3l3 3M22 12l-3-3m3 3l-3 3"/></svg><kbd>W</kbd></button><button data-tool="rotate" title="Rotate (E)" aria-label="Rotate tool (E)"><svg viewBox="0 0 24 24"><path d="M19 8a8 8 0 10.5 7M19 3v5h-5"/></svg><kbd>E</kbd></button><button data-tool="scale" title="Scale (R)" aria-label="Scale tool (R)"><svg viewBox="0 0 24 24"><path d="M5 19L19 5M12 5h7v7M4 15h5v5H4z"/></svg><kbd>R</kbd></button></div><div id="g" data-tool="select"><div class="move-handles"><button class="gizmo-handle gizmo-axis move-axis axis-x" data-kind="move" data-axis="x"><span>X</span></button><button class="gizmo-handle gizmo-axis move-axis axis-y" data-kind="move" data-axis="y"><span>Y</span></button><button class="gizmo-handle gizmo-axis move-axis axis-z" data-kind="move" data-axis="z"><span>Z</span></button></div><svg class="rotate-handles" viewBox="-52 -52 104 104" aria-label="Rotation gizmo"><polyline class="gizmo-handle rotate-ring axis-x" data-kind="rotate" data-axis="x"/><polyline class="gizmo-handle rotate-ring axis-y" data-kind="rotate" data-axis="y"/><polyline class="gizmo-handle rotate-ring axis-z" data-kind="rotate" data-axis="z"/></svg><div class="scale-handles"><button class="gizmo-handle gizmo-axis scale-axis axis-x" data-kind="scale" data-axis="x"><span>X</span></button><button class="gizmo-handle gizmo-axis scale-axis axis-y" data-kind="scale" data-axis="y"><span>Y</span></button><button class="gizmo-handle gizmo-axis scale-axis axis-z" data-kind="scale" data-axis="z"><span>Z</span></button><button class="gizmo-handle scale-uniform" data-kind="scale" data-axis="uniform" title="Uniform scale"></button></div></div><div id="h"><strong>OVRT</strong> · <span id="p">starting</span><br>render <span id="f">0</span> · physics <span id="pf">0</span></div><div id="m" aria-label="Perception view"><button data-mode="rgb" class="on">RGB</button><button data-mode="depth">Depth IR</button><button data-mode="segmentation">Segments</button><button data-mode="detection">Boxes</button><button data-mode="composite">Composite</button></div><div id="help">Q Select · W Move · E Rotate · R Scale · Orbit: left · Pan: middle · Zoom: right/wheel · Alt also supported</div><script>
const v=document.querySelector('#v'),s=document.querySelector('#s'),e=document.querySelector('#e'),g=document.querySelector('#g');let drag=null,pending=null,raf=0,selection=null,gdrag=null,pendingTransform=null,previewTransform=null,transformRaf=0,tool='select';
async function post(path,x){try{return await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(x)})}catch{return null}}
function notify(type,payload){if(window.parent!==window)window.parent.postMessage({type,...payload},'*')}
function imageRect(){const r=v.getBoundingClientRect(),iw=s.naturalWidth||r.width,ih=s.naturalHeight||r.height,k=Math.min(r.width/iw,r.height/ih),width=iw*k,height=ih*k;return{left:r.left+(r.width-width)/2,top:r.top+(r.height-height)/2,width,height}}
function positionGizmo(x,y){const r=imageRect();g.style.left=`${r.left+x*r.width}px`;g.style.top=`${r.top+y*r.height}px`}
function showSelection(value){selection=value&&value.path?value:null;if(!selection||tool==='select'){g.style.display='none';return}const state=selection.gizmo||{};if(!Number.isFinite(state.x)||!Number.isFinite(state.y)){g.style.display='none';return}g.classList.remove('is-picking');g.dataset.tool=tool;positionGizmo(state.x,state.y);g.style.display='block';g.querySelectorAll('.gizmo-axis').forEach(button=>{const isScale=button.classList.contains('scale-axis'),axis=button.dataset.axis,data=(isScale?state.local_axes:state.axes)?.[axis],screen=data?.screen,editable=isScale?selection.scale_editable!==false:selection.editable,enabled=editable&&Array.isArray(screen)&&screen.length===2&&Number.isFinite(data?.pixels_per_meter)&&data.pixels_per_meter>0;button.hidden=!enabled;if(enabled)button.style.transform=`rotate(${Math.atan2(screen[1],screen[0])}rad)`});g.querySelectorAll('.rotate-ring').forEach(ring=>{const points=state.local_axes?.[ring.dataset.axis]?.ring,enabled=selection.editable&&Array.isArray(points)&&points.length>3;ring.style.display=enabled?'':'none';if(enabled)ring.setAttribute('points',points.map(point=>`${point[0]},${point[1]}`).join(' '))});g.querySelector('.scale-uniform').style.display=selection.scale_editable!==false&&selection.editable?'':'none'}
function setTool(next){if(!['select','move','rotate','scale'].includes(next))return;tool=next;g.dataset.tool=tool;post('/api/tool',{tool});document.querySelectorAll('#tools button').forEach(button=>button.classList.toggle('on',button.dataset.tool===tool));if(selection&&!drag&&!gdrag)showSelection(selection)}
function flush(){raf=0;if(!pending)return;const x=pending;pending=null;post('/api/camera',x)}
function move(action,dx,dy){if(action==='zoom'){pending={action,delta:(pending?.delta||0)-(dx+dy)*Math.SQRT1_2}}else if(pending&&pending.action===action){pending.dx+=dx;pending.dy+=dy}else pending={action,dx,dy};if(!raf)raf=requestAnimationFrame(flush)}
async function pick(event){const r=imageRect(),x=(event.clientX-r.left)/r.width,y=(event.clientY-r.top)/r.height;if(x<0||x>1||y<0||y>1)return;g.style.display='none';const response=await post('/api/pick',{x,y});if(!response?.ok){return}const value=await response.json();showSelection(value);notify('blacknode-newton-selection',{path:value.path||''})}
function flushTransform(){transformRaf=0;if(!pendingTransform)return;previewTransform=pendingTransform;pendingTransform=null;post('/api/transform-preview',previewTransform)}
function beginGizmo(event){event.preventDefault();event.stopPropagation();const kind=event.currentTarget.dataset.kind,axis=event.currentTarget.dataset.axis,index={x:0,y:1,z:2}[axis],axes=kind==='move'?selection?.gizmo?.axes:selection?.gizmo?.local_axes,data=axes?.[axis],centerX=Number.parseFloat(g.style.left),centerY=Number.parseFloat(g.style.top);if(!selection?.editable||kind==='move'&&(!data||index===undefined)||kind==='rotate'&&(!data||index===undefined)||kind==='scale'&&(!selection.scale_editable||axis!=='uniform'&&(!data||index===undefined)))return;let screen=data?.screen||[Math.SQRT1_2,-Math.SQRT1_2];if(kind==='rotate'&&Array.isArray(data?.ring)){let nearest=0,best=Infinity;data.ring.forEach((point,i)=>{const distance=Math.hypot(centerX+point[0]-event.clientX,centerY+point[1]-event.clientY);if(distance<best){best=distance;nearest=i}});const before=data.ring[(nearest-1+data.ring.length)%data.ring.length],after=data.ring[(nearest+1)%data.ring.length],length=Math.hypot(after[0]-before[0],after[1]-before[1])||1;screen=[(after[0]-before[0])/length,(after[1]-before[1])/length]}gdrag={kind,axis,index,startX:event.clientX,startY:event.clientY,startLeft:centerX,startTop:centerY,screen,ppm:data?.pixels_per_meter||1,transform:structuredClone(selection.transform)};event.currentTarget.setPointerCapture(event.pointerId)}
function moveGizmo(event){if(!gdrag)return;event.preventDefault();event.stopPropagation();const dx=event.clientX-gdrag.startX,dy=event.clientY-gdrag.startY,transform=structuredClone(gdrag.transform);if(gdrag.kind==='move'){const pixels=dx*gdrag.screen[0]+dy*gdrag.screen[1];transform.translate_m[gdrag.index]+=pixels/gdrag.ppm;g.style.left=`${gdrag.startLeft+gdrag.screen[0]*pixels}px`;g.style.top=`${gdrag.startTop+gdrag.screen[1]*pixels}px`}else if(gdrag.kind==='rotate'){const pixels=dx*gdrag.screen[0]+dy*gdrag.screen[1];transform.rotate_deg[gdrag.index]+=pixels/44*180/Math.PI}else{const pixels=gdrag.axis==='uniform'?(dx-dy)*Math.SQRT1_2:dx*gdrag.screen[0]+dy*gdrag.screen[1],factor=Math.exp(pixels/120);if(gdrag.axis==='uniform')transform.scale=transform.scale.map(value=>Math.max(1e-4,value*factor));else transform.scale[gdrag.index]=Math.max(1e-4,transform.scale[gdrag.index]*factor)}selection={...selection,transform};pendingTransform={path:selection.path,transform};if(!transformRaf)transformRaf=requestAnimationFrame(flushTransform)}
function endGizmo(event){if(!gdrag)return;event.preventDefault();event.stopPropagation();if(transformRaf){cancelAnimationFrame(transformRaf);transformRaf=0}if(pendingTransform){previewTransform=pendingTransform;pendingTransform=null;post('/api/transform-preview',previewTransform)}if(previewTransform)notify('blacknode-newton-transform',previewTransform);previewTransform=null;try{event.currentTarget.releasePointerCapture(event.pointerId)}catch{}gdrag=null}
g.querySelectorAll('.gizmo-handle').forEach(handle=>{handle.addEventListener('pointerdown',beginGizmo);handle.addEventListener('pointermove',moveGizmo);handle.addEventListener('pointerup',endGizmo);handle.addEventListener('pointercancel',endGizmo)});
function end(x){const wasClick=drag&&!drag.alt&&drag.b===0&&drag.distance<4;if(wasClick)pick(x);drag=null;v.classList.remove('drag');if(selection&&!wasClick)showSelection(selection);try{v.releasePointerCapture(x.pointerId)}catch{}}
v.onpointerdown=x=>{if(x.button<0||x.button>2)return;x.preventDefault();drag={x:x.clientX,y:x.clientY,b:x.button,alt:x.altKey,distance:0,navigating:false};v.setPointerCapture(x.pointerId)};v.onpointermove=x=>{if(!drag)return;x.preventDefault();const dx=x.clientX-drag.x,dy=x.clientY-drag.y;drag.x=x.clientX;drag.y=x.clientY;drag.distance+=Math.hypot(dx,dy);if(drag.distance<3)return;if(!drag.navigating){drag.navigating=true;v.classList.add('drag');g.style.display='none'}move(drag.b===0?'orbit':drag.b===1?'pan':'zoom',dx,dy)};v.onpointerup=end;v.onpointercancel=end;v.onlostpointercapture=()=>{drag=null;v.classList.remove('drag');if(selection&&!gdrag)showSelection(selection)};v.ondragstart=x=>x.preventDefault();v.oncontextmenu=x=>x.preventDefault();v.onwheel=x=>{x.preventDefault();post('/api/camera',{action:'zoom',delta:x.deltaY})};v.ondblclick=()=>post('/api/camera',{action:'reset'});
async function st(){try{const x=await(await fetch('/api/status',{cache:'no-store'})).json();document.querySelector('#p').textContent=x.error?'error':x.phase;document.querySelector('#f').textContent=x.frame;document.querySelector('#pf').textContent=x.physics_frame;document.querySelector('#s1').textContent=x.error?'OVRT render failed':'Starting NVIDIA OVRT';document.querySelector('#d').textContent=x.error||x.detail;if(x.has_image)e.style.display='none';if(!drag&&!gdrag&&x.selection)showSelection(x.selection)}catch{document.querySelector('#p').textContent='disconnected'}}setInterval(st,250);st();window.addEventListener('resize',()=>{if(selection&&!gdrag)showSelection(selection)});
document.querySelectorAll('#tools button').forEach(button=>button.onclick=()=>setTool(button.dataset.tool));window.addEventListener('keydown',event=>{if(event.ctrlKey||event.metaKey||event.altKey)return;const next={q:'select',w:'move',e:'rotate',r:'scale'}[event.key.toLowerCase()];if(!next)return;event.preventDefault();setTool(next)});
const viewModes=new Set(['rgb','depth','segmentation','detection','composite']);
async function setViewMode(mode){if(!viewModes.has(mode))return;const response=await post('/api/view',{mode});if(!response?.ok)return;document.querySelectorAll('#m button').forEach(button=>button.classList.toggle('on',button.dataset.mode===mode));notify('blacknode-newton-view-state',{mode})}
document.querySelectorAll('#m button').forEach(button=>button.onclick=()=>setViewMode(button.dataset.mode));
window.addEventListener('message',event=>{if(event.source!==window.parent)return;const message=event.data;if(message?.type==='blacknode-newton-view')setViewMode(String(message.mode||''))});
</script></body></html>"""


def _loopback_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def _select_viewer_port(requested: int) -> int:
    requested = max(1024, min(65535, int(requested)))
    if _loopback_port_available(requested):
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _render_resolution(config: dict[str, Any]) -> tuple[int, int]:
    """Use a full-HD default while retaining explicit bounded overrides."""
    return (
        max(320, min(3840, int(config.get("width") or 1920))),
        max(240, min(2160, int(config.get("height") or 1080))),
    )


def _source_stage_info(asset_path: str) -> dict[str, Any]:
    """Inspect render-space metadata without making OVRT depend on Newton internals."""
    info: dict[str, Any] = {
        "meters_per_unit": 1.0,
        "up_axis": "z",
        "bounds_min": [],
        "bounds_max": [],
    }
    if not asset_path:
        return info
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return info
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if math.isfinite(meters_per_unit) and meters_per_unit > 0.0:
            info["meters_per_unit"] = meters_per_unit
        up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z").lower()
        if up_axis in {"y", "z"}:
            info["up_axis"] = up_axis
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        if not bounds.IsEmpty():
            lower = [float(value) for value in bounds.GetMin()]
            upper = [float(value) for value in bounds.GetMax()]
            if all(math.isfinite(value) for value in lower + upper):
                info["bounds_min"] = lower
                info["bounds_max"] = upper
    except Exception:
        # Scene construction already reports malformed USD. Camera inspection is
        # best-effort so an unusual boundable can still reach the renderer.
        pass
    return info


def _scene_camera(
    session: Any,
    config: dict[str, Any],
    stage_info: dict[str, Any],
) -> tuple[list[float], list[float], str]:
    camera = dict(config.get("camera") or {})
    position = [float(value) for value in list(camera.get("position_m") or [])]
    target = [float(value) for value in list(camera.get("target_m") or [])]
    up_axis = str(camera.get("up_axis") or "auto").lower()
    source_up_axis = str(stage_info.get("up_axis") or "z")
    meters_per_unit = max(1.0e-12, float(stage_info.get("meters_per_unit") or 1.0))
    if len(position) == 3 and len(target) == 3:
        return (
            [value / meters_per_unit for value in position],
            [value / meters_per_unit for value in target],
            source_up_axis if up_axis == "auto" else up_axis,
        )

    lower = list(stage_info.get("bounds_min") or [])
    upper = list(stage_info.get("bounds_max") or [])
    if len(lower) == 3 and len(upper) == 3:
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
        extent = max(upper[axis] - lower[axis] for axis in range(3))
        distance = max(0.8 / meters_per_unit, extent * 1.8)
        if source_up_axis == "y":
            return (
                [center[0] + distance, center[1] + distance * 0.75, center[2] + distance],
                center,
                "y",
            )
        return (
            [center[0] + distance, center[1] - distance, center[2] + distance * 0.75],
            center,
            "z",
        )

    points: list[list[float]] = []
    try:
        for transform in session.state_0.body_q.numpy():
            point = [float(value) for value in transform[:3]]
            if all(math.isfinite(value) for value in point):
                points.append(point)
    except Exception:
        pass
    if not points:
        center = [0.0, 0.0, 0.15]
        extent = 0.5
    else:
        lower = [min(point[axis] for point in points) for axis in range(3)]
        upper = [max(point[axis] for point in points) for axis in range(3)]
        center = [(lower[axis] + upper[axis]) * 0.5 for axis in range(3)]
        extent = max(0.4, max(upper[axis] - lower[axis] for axis in range(3)))
    distance = max(0.8, extent * 2.2)
    position = [center[0] + distance, center[1] - distance, center[2] + distance * 0.75]
    return (
        [value / meters_per_unit for value in position],
        [value / meters_per_unit for value in center],
        source_up_axis,
    )


def _usd_body_render_frames(
    asset_path: str, body_paths: dict[str, int]
) -> dict[int, dict[str, Any]]:
    """Describe each USD body's authored render parent and inherited scale."""
    if not asset_path or not body_paths:
        return {}
    try:
        import numpy as np
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return {}
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        worlds: dict[int, Any] = {}
        scales: dict[int, list[float]] = {}
        prims: dict[int, Any] = {}
        for path, index in body_paths.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                continue
            world = cache.GetLocalToWorldTransform(prim)
            prims[index] = prim
            worlds[index] = np.asarray(world, dtype=np.float64).reshape(4, 4)
            scales[index] = [float(value) for value in Gf.Transform(world).GetScale()]
        frames: dict[int, dict[str, Any]] = {}
        for index, prim in prims.items():
            parent = prim.GetParent()
            if parent and parent.IsValid() and parent.IsA(UsdGeom.Xformable):
                parent_world = np.asarray(
                    cache.GetLocalToWorldTransform(parent), dtype=np.float64
                ).reshape(4, 4)
            else:
                parent_world = np.eye(4, dtype=np.float64)
            ancestor = parent
            render_parent_index = -1
            while ancestor and ancestor.IsValid():
                candidate = body_paths.get(str(ancestor.GetPath()))
                if candidate is not None and candidate in worlds:
                    render_parent_index = candidate
                    break
                ancestor = ancestor.GetParent()
            if render_parent_index >= 0:
                parent_relative = parent_world @ np.linalg.inv(
                    worlds[render_parent_index]
                )
                render_parent_scale = scales[render_parent_index]
            else:
                parent_relative = parent_world
                render_parent_scale = [1.0, 1.0, 1.0]
            frames[index] = {
                "world_scale": scales[index],
                "render_parent_index": render_parent_index,
                "render_parent_relative": parent_relative.reshape(-1).tolist(),
                "render_parent_world_scale": render_parent_scale,
            }
        return frames
    except Exception:
        return {}


def _usd_interaction_frames(
    asset_path: str, scene_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Capture authored local/parent frames used by the viewport move gizmo."""
    frames: list[dict[str, Any]] = []
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path) if asset_path else None
        cache = UsdGeom.XformCache(Usd.TimeCode.Default()) if stage is not None else None
        for item in scene_items:
            render_path = str(item.get("path") or "")
            path = render_path
            if not render_path or not path or not bool(item.get("editable")):
                continue
            if path == "/Blacknode/Ground":
                frames.append({
                    "path": path,
                    "render_path": "/BlacknodeOVRT/Ground",
                    "parent_world": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
                    "local_matrix": np.eye(4, dtype=np.float64).reshape(-1).tolist(),
                })
                continue
            if stage is None or cache is None:
                continue
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                continue
            local = UsdGeom.Xformable(prim).GetLocalTransformation()
            if isinstance(local, tuple):
                local = local[0]
            parent = prim.GetParent()
            parent_world = (
                cache.GetLocalToWorldTransform(parent)
                if parent and parent.IsValid() and parent.IsA(UsdGeom.Xformable)
                else np.eye(4, dtype=np.float64)
            )
            frames.append({
                "path": path,
                "render_path": render_path,
                "parent_world": np.asarray(parent_world, dtype=np.float64).reshape(-1).tolist(),
                "local_matrix": np.asarray(local, dtype=np.float64).reshape(-1).tolist(),
                "physics_body_index": (
                    int(item.get("physics_body_index", -1))
                    if bool(item.get("physics_pose_editable"))
                    else -1
                ),
                "scale_editable": bool(
                    item.get("editable") and not item.get("physics_pose_editable")
                ),
            })
    except Exception:
        return frames
    return frames


def _body_entries(
    session: Any, model: Any, asset_path: str = ""
) -> list[dict[str, Any]]:
    if str(session.scene.get("asset_format") or "usd").lower() in {"urdf", "mjcf"}:
        try:
            shape_bodies = model.shape_body.numpy().tolist()
            shape_transforms = model.shape_transform.numpy().tolist()
        except Exception:
            return []
        entries: list[dict[str, Any]] = []
        for render_entry in list(getattr(session, "render_shapes", []) or []):
            if not render_entry.get("visual"):
                continue
            shape_index = int(render_entry.get("shape_index", -1))
            if shape_index < 0 or shape_index >= len(shape_bodies):
                continue
            body_index = int(shape_bodies[shape_index])
            if body_index < 0:
                continue
            entries.append({
                "index": body_index,
                "parent_index": -1,
                "path": str(render_entry.get("path") or ""),
                "name": str(model.shape_label[shape_index] or f"shape_{shape_index}"),
                "scale": [1.0, 1.0, 1.0],
                "local_pose": [float(value) for value in shape_transforms[shape_index]],
            })
        # Newton's generated USD already contains the authoritative rendered
        # bind pose. In particular, MJCF mesh compiler transforms are not
        # always identical to model.shape_transform. Preserve the authored
        # shape/body relationship so the first streamed physics pose cannot
        # scatter visual pieces that were correct in the loaded frame.
        try:
            import numpy as np
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(asset_path) if asset_path else None
            state = getattr(session, "state_0", None)
            transforms = state.body_q.numpy().tolist() if state is not None else []
            if stage is None or not transforms:
                return entries
            meters_per_unit = max(1.0e-12, float(UsdGeom.GetStageMetersPerUnit(stage)))
            cache = UsdGeom.XformCache(Usd.TimeCode(0.0))
            for entry in entries:
                body_index = int(entry["index"])
                prim = stage.GetPrimAtPath(str(entry["path"]))
                if (
                    body_index >= len(transforms)
                    or not prim
                    or not prim.IsValid()
                    or not prim.IsA(UsdGeom.Xformable)
                ):
                    continue
                parent = prim.GetParent()
                parent_world = (
                    cache.GetLocalToWorldTransform(parent)
                    if parent and parent.IsValid() and parent.IsA(UsdGeom.Xformable)
                    else np.eye(4, dtype=np.float64)
                )
                entry["body_index"] = body_index
                entry["initial_world_matrix"] = np.asarray(
                    cache.GetLocalToWorldTransform(prim), dtype=np.float64
                ).reshape(-1).tolist()
                entry["body_bind_pose"] = [
                    float(value) for value in transforms[body_index]
                ]
                entry["render_parent_world_matrix"] = np.asarray(
                    parent_world, dtype=np.float64
                ).reshape(-1).tolist()
        except Exception:
            # Retain the older shape-local mapping as a compatibility fallback
            # for minimal USD installations or incomplete generated assets.
            pass
        return entries
    runtime_bodies = {
        str(body.get("name") or ""): body
        for body in list(session.scene.get("rigid_bodies") or [])
    }
    entries: list[dict[str, Any]] = []
    parent_by_body: dict[int, int] = {}
    try:
        for parent, child in zip(
            model.joint_parent.numpy().tolist(), model.joint_child.numpy().tolist()
        ):
            parent_by_body[int(child)] = int(parent)
    except Exception:
        parent_by_body = {}
    for index, raw_label in enumerate(list(model.body_label)):
        label = str(raw_label or "")
        if label.startswith("/"):
            # OVStage does not reliably propagate live writes on container
            # Xforms through a populated USD hierarchy. USD Gprims are updated
            # directly from their authored body-relative frames instead.
            continue
        body = runtime_bodies.get(label)
        if body is None:
            continue
        entries.append({
            "index": index,
            "parent_index": parent_by_body.get(index, -1),
            "path": f"/BlacknodeOVRT/RigidBodies/body_{index}",
            "name": label,
            "scale": [float(value) for value in body.get("size_m", [0.05, 0.05, 0.05])],
            "scale_in_meters": True,
            "color": [float(value) for value in body.get("color_rgb", [0.8, 0.25, 0.12])],
        })
    render_frames = _usd_body_render_frames(
        asset_path,
        {
            str(entry["path"]): int(entry["index"])
            for entry in entries
            if str(entry.get("path") or "").startswith("/")
        },
    )
    for entry in entries:
        entry.update(render_frames.get(int(entry["index"]), {}))
    return entries


def _collision_wireframes(
    asset_path: str, render_shapes: list[dict[str, Any]], max_edges_per_shape: int = 600
) -> list[dict[str, Any]]:
    """Extract bounded collision edges for a depth-independent viewport overlay."""
    if not asset_path:
        return []
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return []
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        box_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        result: list[dict[str, Any]] = []
        for entry in render_shapes:
            if not entry.get("collider"):
                continue
            source_path = str(entry.get("source_path") or entry.get("path") or "")
            prim = stage.GetPrimAtPath(source_path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
                continue
            points: Any
            edges: list[tuple[int, int]]
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
                counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
                indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
                edge_set: set[tuple[int, int]] = set()
                cursor = 0
                for count in counts:
                    face = [int(value) for value in indices[cursor:cursor + int(count)]]
                    cursor += int(count)
                    for offset, start in enumerate(face):
                        end = face[(offset + 1) % len(face)]
                        if start != end:
                            edge_set.add((min(start, end), max(start, end)))
                edges = sorted(edge_set)
            else:
                bounds = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                ).ComputeLocalBound(prim).ComputeAlignedRange()
                if bounds.IsEmpty():
                    continue
                lower = np.asarray(bounds.GetMin(), dtype=np.float64)
                upper = np.asarray(bounds.GetMax(), dtype=np.float64)
                points = np.asarray([
                    [x, y, z]
                    for z in (lower[2], upper[2])
                    for y in (lower[1], upper[1])
                    for x in (lower[0], upper[0])
                ], dtype=np.float64)
                edges = box_edges
            if points.ndim != 2 or points.shape[1] != 3 or not edges:
                continue
            limit = max(12, int(max_edges_per_shape))
            if len(edges) > limit:
                stride = len(edges) / float(limit)
                edges = [edges[min(len(edges) - 1, int(index * stride))] for index in range(limit)]
            used = sorted({vertex for edge in edges for vertex in edge})
            if not used or max(used) >= len(points):
                continue
            remap = {old: new for new, old in enumerate(used)}
            compact_points = points[used]
            source_world = np.asarray(
                cache.GetLocalToWorldTransform(prim), dtype=np.float64
            ).reshape(4, 4)
            homogeneous = np.column_stack((compact_points, np.ones(len(compact_points))))
            bind_world_points = (homogeneous @ source_world)[:, :3]
            wireframe = {
                "path": str(entry.get("path") or ""),
                "source_path": source_path,
                "points_bind_world": bind_world_points.tolist(),
                "edges": [[remap[start], remap[end]] for start, end in edges],
                "body_index": int(entry.get("body_index", -1)),
                "body_bind_world_matrix": list(entry.get("body_bind_world_matrix") or []),
            }
            result.append(wireframe)
        return result
    except Exception:
        return []


def _ghost_entries(session: Any, model: Any, asset_path: str) -> list[dict[str, Any]]:
    """Describe visual robot shapes that can be instanced as the reference ghost."""
    if not asset_path:
        return []
    try:
        import newton
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return []
        robot_bodies = set(getattr(session, "articulation_body_indices", set()) or set())
        shape_bodies = model.shape_body.numpy().tolist()
        shape_flags = model.shape_flags.numpy().tolist()
        shape_transforms = model.shape_transform.numpy().tolist()
        shape_scales = model.shape_scale.numpy().tolist()
        generated_model = str(
            session.scene.get("asset_format") or "usd"
        ).lower() in {"urdf", "mjcf"}
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for shape_index, body_index in enumerate(shape_bodies):
            body_index = int(body_index)
            if body_index not in robot_bodies:
                continue
            if not int(shape_flags[shape_index]) & int(newton.ShapeFlags.VISIBLE):
                continue
            source_path = (
                f"/root/model/shapes/shape_{shape_index}/instance_0"
                if generated_model
                else str(model.shape_label[shape_index] or "")
            )
            prim = stage.GetPrimAtPath(source_path) if source_path.startswith("/") else None
            if (not prim or not prim.IsValid()) and source_path.endswith("_visual"):
                source_path = source_path[:-7]
                prim = stage.GetPrimAtPath(source_path)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Subset):
                prim = prim.GetParent()
                source_path = str(prim.GetPath())
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
                continue
            key = (source_path, body_index)
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "index": body_index,
                "shape_index": shape_index,
                "source_path": source_path,
                "path": f"/BlacknodeOVRT/RealGhost/shape_{shape_index}",
                "scale": [float(value) for value in shape_scales[shape_index]],
                "local_pose": [float(value) for value in shape_transforms[shape_index]],
                "parent_index": -1,
            })
        return entries
    except Exception:
        return []


def _state_transforms(state: Any) -> list[list[float]]:
    """Return Newton body poses while treating a valid empty stage as empty."""
    body_q = getattr(state, "body_q", None)
    if body_q is None:
        return []
    return body_q.numpy().tolist()


def _usd_bound_render_shapes(model: Any, asset_path: str) -> list[dict[str, Any]]:
    """Build live visual bindings when a lightweight session has no scene inventory."""
    if not asset_path:
        return []
    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return []
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        body_paths = sorted(
            (
                (str(label), index)
                for index, label in enumerate(model.body_label)
                if str(label).startswith("/")
                and stage.GetPrimAtPath(str(label)).IsValid()
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        body_bind_worlds = {
            index: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            for path, index in body_paths
        }
        entries: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Gprim):
                continue
            path = str(prim.GetPath())
            body_index = next(
                (
                    index
                    for body_path, index in body_paths
                    if path == body_path or path.startswith(body_path + "/")
                ),
                -1,
            )
            if body_index < 0:
                continue
            lowered = path.lower()
            collider = "/collisions/" in lowered or "/collision/" in lowered
            shape_bind_world = cache.GetLocalToWorldTransform(prim)
            parent = prim.GetParent()
            render_parent_world = (
                cache.GetLocalToWorldTransform(parent)
                if parent and parent.IsValid() and parent.IsA(UsdGeom.Xformable)
                else Gf.Matrix4d(1.0)
            )
            entries.append({
                "path": path,
                "source_path": path,
                "visual": not collider,
                "collider": collider,
                "body_index": body_index,
                "initial_world_matrix": [
                    float(shape_bind_world[row][column])
                    for row in range(4) for column in range(4)
                ],
                "body_bind_world_matrix": [
                    float(body_bind_worlds[body_index][row][column])
                    for row in range(4) for column in range(4)
                ],
                "render_parent_world_matrix": [
                    float(render_parent_world[row][column])
                    for row in range(4) for column in range(4)
                ],
            })
        return entries
    except Exception:
        return []


class OVRTViewer:
    """Proxy a live Newton session to an isolated OVRT/OVStage renderer."""

    def __init__(self, session: Any, model: Any, config: dict[str, Any]) -> None:
        self.session = session
        self.requested_port = int(config.get("port") or 8080)
        self.port = _select_viewer_port(self.requested_port)
        self._closed = False
        self._ready = threading.Event()
        self._last_error = ""
        self._diagnostics: list[str] = []
        self._web_lock = threading.RLock()
        self._frame_ready = threading.Condition(self._web_lock)
        self._jpeg = b""
        self._render_frame = 0
        self._worker_collision_wireframe_count = 0
        self._worker_colliders_visible = False
        self._worker_collision_overlay_pixels = 0
        self._worker_collision_overlay_segments = 0
        self._worker_collision_depth_range = [0.0, 0.0]
        self._physics_frame = 0
        self._web_phase = "starting"
        self._web_detail = "Starting isolated OVRT render worker"
        self._view_mode = "rgb"
        self.selected_path = ""
        self._selection_gizmo: dict[str, Any] = {}
        self._selection_version = 0
        self._selection_ready = threading.Condition(self._web_lock)
        self._sender_started = False
        self._updates: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=256)

        self._http_server = self._start_http_server()
        self._protocol_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._protocol_listener.bind(("127.0.0.1", 0))
        self._protocol_listener.listen(1)
        self._protocol_listener.settimeout(15.0)
        self._protocol_socket: socket.socket | None = None

        asset_path = str(getattr(session, "render_asset_path", "") or session.scene.get("asset_path") or "")
        stage_info = _source_stage_info(asset_path)
        position, target, up_axis = _scene_camera(session, config, stage_info)
        self._up_axis = up_axis
        width, height = _render_resolution(config)
        meters_per_unit = float(stage_info["meters_per_unit"])
        self._meters_per_unit = meters_per_unit
        bounds_min = list(stage_info.get("bounds_min") or [])
        bounds_max = list(stage_info.get("bounds_max") or [])
        bounds_extent = (
            max(bounds_max[axis] - bounds_min[axis] for axis in range(3))
            if len(bounds_min) == 3 and len(bounds_max) == 3
            else 0.0
        )
        workspace_edits = dict(session.scene.get("workspace_edits") or {})
        try:
            initial_body_transforms = [
                [float(value) for value in transform]
                for transform in session.state_0.body_q.numpy().tolist()
            ]
        except Exception:
            initial_body_transforms = []
        render_shapes = list(getattr(session, "render_shapes", []) or [])
        if not render_shapes:
            render_shapes = _usd_bound_render_shapes(model, asset_path)
        collision_wireframes = _collision_wireframes(asset_path, render_shapes)
        self._collision_wireframe_count = len(collision_wireframes)
        worker_config = {
            "port": self.port,
            "label": str(config.get("label") or "Blacknode Newton · OVRT"),
            "asset_path": asset_path,
            "ground_enabled": bool(session.scene.get("ground", {}).get("enabled", True)),
            "ground_height": float(session.scene.get("ground", {}).get("height_m", 0.0)),
            "meters_per_unit": meters_per_unit,
            "up_axis": up_axis,
            "grid_extent": max(1.0 / meters_per_unit, bounds_extent * 0.6),
            "show_grid": bool(config.get("show_grid", True)),
            "background_color": str(config.get("background_color") or "#6383c5"),
            "environment": dict(config.get("environment") or {}),
            "ground_transform": dict(
                dict(workspace_edits.get("transforms") or {}).get("/Blacknode/Ground") or {}
            ),
            "ground_material": dict(
                dict(workspace_edits.get("materials") or {}).get("/Blacknode/Ground") or {}
            ),
            "body_entries": _body_entries(session, model, asset_path),
            "initial_body_transforms": initial_body_transforms,
            "ghost_entries": _ghost_entries(session, model, asset_path),
            "render_shapes": render_shapes,
            "collision_wireframes": collision_wireframes,
            "interaction_frames": _usd_interaction_frames(
                asset_path, list(getattr(session, "scene_items", []) or [])
            ),
            "visibility_overrides": dict(workspace_edits.get("visibility") or {}),
            "show_visuals": bool(
                getattr(session, "show_visuals", config.get("show_visuals", True))
            ),
            "show_colliders": bool(
                getattr(session, "show_colliders", config.get("show_colliders", False))
            ),
            "camera": {"position": position, "target": target, "up_axis": up_axis},
            "width": width,
            "height": height,
            "render_fps": max(1, min(60, int(config.get("render_fps") or 60))),
            # OVRT may return no LdrColor while shaders are compiling. The worker
            # treats this as a maximum readiness budget and exits early on the
            # first usable frame.
            "warmup_frames": max(1, min(120, int(config.get("warmup_frames", 40)))),
            "jpeg_quality": max(50, min(98, int(config.get("jpeg_quality") or 90))),
            "http_in_parent": True,
            "protocol_port": int(self._protocol_listener.getsockname()[1]),
        }
        worker_path = Path(__file__).with_name("worker.py")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                creationflags=0,
            )
        except Exception:
            self._closed = True
            self._protocol_listener.close()
            self._http_server.shutdown()
            self._http_server.server_close()
            raise
        self._protocol_thread = threading.Thread(target=self._read_protocol, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._sender_thread = threading.Thread(target=self._send_updates, daemon=True)
        self._protocol_thread.start()
        self._stderr_thread.start()
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps({"type": "configure", "config": worker_config}) + "\n")
        self._process.stdin.flush()
        if not self._ready.wait(10.0):
            try:
                self._raise_if_exited()
                detail = "OVRT viewer worker did not establish its control channel within 10 seconds"
            except RuntimeError as exc:
                detail = str(exc)
            self.close()
            raise RuntimeError(detail)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read_protocol(self) -> None:
        try:
            connection, _address = self._protocol_listener.accept()
            self._protocol_socket = connection
        except OSError as exc:
            if not self._closed:
                self._last_error = f"OVRT protocol connection failed: {exc}"
            return
        with connection, connection.makefile("r", encoding="utf-8") as stream:
            for raw_line in stream:
                self._handle_worker_event(raw_line)

    def _handle_worker_event(self, raw_line: str) -> None:
        line = raw_line.strip()
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            if line:
                self._diagnostics.append(line)
            return
        if event.get("type") == "ready":
            with self._web_lock:
                self._web_phase = "initializing"
                self._web_detail = "Creating NVIDIA OVRT renderer; first launch compiles RTX shaders"
            self._ready.set()
        elif event.get("type") == "status":
            with self._web_lock:
                self._web_phase = str(event.get("phase") or "initializing")
                self._web_detail = str(event.get("detail") or "Starting NVIDIA OVRT")
                if self._web_phase in {"loading", "warming", "streaming"} and not self._sender_started:
                    self._sender_started = True
                    self._sender_thread.start()
        elif event.get("type") == "frame":
            try:
                frame = base64.b64decode(str(event.get("jpeg") or ""), validate=True)
            except (ValueError, TypeError):
                return
            with self._frame_ready:
                self._jpeg = frame
                self._render_frame += 1
                self._web_phase = "streaming"
                self._web_detail = "OVRT RTX render stream"
                self._worker_collision_wireframe_count = int(
                    event.get("collision_wireframe_count") or 0
                )
                self._worker_colliders_visible = bool(event.get("colliders_visible"))
                self._worker_collision_overlay_pixels = int(
                    event.get("collision_overlay_pixels") or 0
                )
                self._worker_collision_overlay_segments = int(
                    event.get("collision_overlay_segments") or 0
                )
                self._worker_collision_depth_range = list(
                    event.get("collision_depth_range") or [0.0, 0.0]
                )
                self._frame_ready.notify_all()
        elif event.get("type") == "error":
            self._last_error = str(event.get("message") or "OVRT worker failed")
            with self._frame_ready:
                self._web_phase = "error"
                self._web_detail = "OVRT worker failed"
                self._frame_ready.notify_all()
        elif event.get("type") == "selection":
            with self._selection_ready:
                self.selected_path = _workspace_scene_path(event.get("path"))
                self._selection_gizmo = dict(event.get("gizmo") or {})
                self._selection_version += 1
                self._selection_ready.notify_all()
        elif event.get("type") == "gizmo":
            with self._web_lock:
                if str(event.get("path") or "") == self.selected_path:
                    self._selection_gizmo = dict(event.get("gizmo") or {})

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        pending = ""
        while True:
            chunk = self._process.stderr.read(4096)
            if not chunk:
                break
            pending += chunk
            lines = pending.splitlines(keepends=True)
            pending = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                pending = lines.pop()
            for line in lines:
                clean = line.strip()
                if clean:
                    self._diagnostics.append(clean)
                    del self._diagnostics[:-50]
        if pending.strip():
            self._diagnostics.append(pending.strip())

    def _send_updates(self) -> None:
        while True:
            update = self._updates.get()
            if update is None:
                return
            try:
                if self._process.stdin is None:
                    return
                self._process.stdin.write(json.dumps(update, separators=(",", ":")) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def _raise_if_exited(self) -> None:
        code = self._process.poll()
        if code is None:
            return
        detail = self._last_error or (self._diagnostics[-1] if self._diagnostics else "no diagnostics")
        raise RuntimeError(f"OVRT viewer worker exited with code {code}: {detail}")

    def _selection_payload(self) -> dict[str, Any]:
        path = str(self.selected_path or "")
        item = next(
            (
                value for value in list(getattr(self.session, "scene_items", []) or [])
                if str(value.get("path") or "") == path
            ),
            None,
        )
        return {
            "path": path,
            "name": str((item or {}).get("name") or path.rsplit("/", 1)[-1]),
            "editable": bool((item or {}).get("editable")),
            "scale_editable": bool(
                (item or {}).get("editable")
                and not (item or {}).get("physics_pose_editable")
            ),
            "transform": dict((item or {}).get("transform") or {}),
            "gizmo": dict(self._selection_gizmo),
        }

    def _wait_for_selection(self, version: int, timeout: float = 1.5) -> dict[str, Any]:
        with self._selection_ready:
            self._selection_ready.wait_for(
                lambda: self._selection_version != version or self._closed,
                timeout=timeout,
            )
            return self._selection_payload()

    def _start_http_server(self) -> ThreadingHTTPServer:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send_headers(self, status: int, content_type: str, length: int | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    body = _VIEWER_HTML.encode("utf-8")
                    self._send_headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
                    self.wfile.write(body)
                    return
                if path == "/api/status":
                    with viewer._web_lock:
                        status = {
                            "phase": viewer._web_phase,
                            "error": viewer._last_error,
                            "detail": viewer._web_detail,
                            "frame": viewer._render_frame,
                            "physics_frame": viewer._physics_frame,
                            "view_mode": viewer._view_mode,
                            "has_image": bool(viewer._jpeg),
                            "selection": viewer._selection_payload(),
                        }
                    body = json.dumps(status).encode("utf-8")
                    self._send_headers(HTTPStatus.OK, "application/json", len(body))
                    self.wfile.write(body)
                    return
                if path == "/stream.mjpg":
                    self._send_headers(HTTPStatus.OK, "multipart/x-mixed-replace; boundary=frame")
                    seen = -1
                    try:
                        while not viewer._closed:
                            with viewer._frame_ready:
                                viewer._frame_ready.wait_for(
                                    lambda: viewer._render_frame != seen or viewer._closed, timeout=2.0
                                )
                                if viewer._closed:
                                    return
                                seen = viewer._render_frame
                                frame = viewer._jpeg
                            if not frame:
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
                            )
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    return
                self._send_headers(HTTPStatus.NOT_FOUND, "text/plain", 0)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path not in {
                    "/api/camera", "/api/view", "/api/pick", "/api/tool", "/api/transform-preview"
                }:
                    self._send_headers(HTTPStatus.NOT_FOUND, "text/plain", 0)
                    return
                try:
                    length = min(16384, int(self.headers.get("Content-Length") or 0))
                    value = json.loads(self.rfile.read(length) or b"{}")
                    if path == "/api/pick":
                        x = float(value.get("x"))
                        y = float(value.get("y"))
                        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                            raise ValueError("pick coordinates must be normalized")
                        with viewer._web_lock:
                            selection_version = viewer._selection_version
                        viewer._queue_update({"type": "pick", "x": x, "y": y})
                        payload = viewer._wait_for_selection(selection_version)
                        body = json.dumps(payload).encode("utf-8")
                        self._send_headers(HTTPStatus.OK, "application/json", len(body))
                        self.wfile.write(body)
                        return
                    if path == "/api/view":
                        mode = str(value.get("mode") or "rgb").lower()
                        if mode not in {"rgb", "depth", "segmentation", "detection", "composite"}:
                            raise ValueError("unsupported perception view")
                        with viewer._web_lock:
                            viewer._view_mode = mode
                        viewer._queue_update({"type": "view", "mode": mode})
                        self._send_headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                        return
                    if path == "/api/tool":
                        tool = str(value.get("tool") or "select").lower()
                        if tool not in {"select", "move", "rotate", "scale"}:
                            raise ValueError("unsupported viewport tool")
                        viewer._queue_update({"type": "gizmo_tool", "tool": tool})
                        self._send_headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                        return
                    if path == "/api/transform-preview":
                        transform = dict(value.get("transform") or {})
                        clean_transform: dict[str, list[float]] = {}
                        for name, default in (
                            ("translate_m", [0.0, 0.0, 0.0]),
                            ("rotate_deg", [0.0, 0.0, 0.0]),
                            ("scale", [1.0, 1.0, 1.0]),
                        ):
                            raw = list(transform.get(name) or default)
                            if len(raw) != 3:
                                raise ValueError("transform vectors require three values")
                            clean_transform[name] = [float(item) for item in raw]
                            if not all(math.isfinite(item) for item in clean_transform[name]):
                                raise ValueError("transform values must be finite")
                        if any(item <= 0.0 for item in clean_transform["scale"]):
                            raise ValueError("transform scale must be positive")
                        preview_path = str(value.get("path") or "")
                        if not preview_path.startswith("/"):
                            raise ValueError("transform preview requires an absolute prim path")
                        viewer._queue_update({
                            "type": "transform",
                            "path": preview_path,
                            "transform": clean_transform,
                            "meters_per_unit": viewer._meters_per_unit,
                            "preview": True,
                        })
                        self._send_headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                        return
                    action = str(value.get("action") or "")
                    if action not in {"orbit", "pan", "zoom", "reset"}:
                        raise ValueError("unsupported camera action")
                    event: dict[str, Any] = {"type": "camera", "action": action}
                    for name in ("dx", "dy", "delta"):
                        if name in value:
                            event[name] = max(-500.0, min(500.0, float(value[name])))
                    viewer._queue_update(event)
                    self._send_headers(HTTPStatus.NO_CONTENT, "text/plain", 0)
                except (ValueError, TypeError, json.JSONDecodeError):
                    self._send_headers(HTTPStatus.BAD_REQUEST, "text/plain", 0)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = False

        server = Server(("127.0.0.1", self.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True, name="ovrtx-proxy-http").start()
        return server

    def _queue_update(self, update: dict[str, Any]) -> None:
        try:
            self._updates.put_nowait(update)
        except queue.Full:
            # Pose frames are replaceable. Never evict a queued live-edit or
            # camera command merely to enqueue a newer pose.
            if update.get("type") in {"poses", "reference_pose"}:
                return
            try:
                self._updates.get_nowait()
            except queue.Empty:
                pass
            try:
                self._updates.put_nowait(update)
            except queue.Full:
                pass

    def is_running(self) -> bool:
        if self._closed:
            return False
        self._raise_if_exited()
        return True

    def begin_frame(self, time_seconds: float) -> None:
        del time_seconds

    def log_state(self, state: Any) -> None:
        self._raise_if_exited()
        transforms = _state_transforms(state)
        update = {"type": "poses", "transforms": transforms, "frame": self.session.frame_count}
        with self._web_lock:
            self._physics_frame = int(self.session.frame_count)
        self._queue_update(update)

    def log_reference_state(self, state: Any | None, options: dict[str, Any]) -> None:
        self._raise_if_exited()
        self._queue_update({
            "type": "reference_pose",
            "transforms": _state_transforms(state) if state is not None else [],
            "visible": bool(state is not None and options.get("visible", True)),
            "offset_m": [float(value) for value in list(options.get("offset_m") or [0, 0, 0])],
        })

    def end_frame(self) -> None:
        pass

    def set_visibility(self, path: str, visible: bool) -> bool:
        self._raise_if_exited()
        self._queue_update({
            "type": "visibility",
            "path": str(path),
            "visible": bool(visible),
        })
        return True

    def set_selection(self, path: str) -> bool:
        self._raise_if_exited()
        self.selected_path = _workspace_scene_path(path)
        self._queue_update({"type": "select", "path": self.selected_path})
        return True

    def set_grid(self, visible: bool) -> bool:
        self._raise_if_exited()
        self._queue_update({"type": "grid", "visible": bool(visible)})
        return True

    def set_render_options(self, show_visuals: bool, show_colliders: bool) -> bool:
        self._raise_if_exited()
        self._queue_update({
            "type": "render_options",
            "show_visuals": bool(show_visuals),
            "show_colliders": bool(show_colliders),
        })
        return True

    def set_transform(self, path: str, transform: dict[str, Any]) -> bool:
        self._raise_if_exited()
        clean = dict(transform)
        physics_backed = bool(clean.pop("_physics_backed", False))
        self._queue_update({
            "type": "transform", "path": str(path), "transform": clean,
            "physics_backed": physics_backed,
            "meters_per_unit": self._meters_per_unit,
        })
        return True

    def set_material(self, path: str, material_path: str, material: dict[str, Any]) -> bool:
        self._raise_if_exited()
        self._queue_update({
            "type": "material", "path": str(path), "material_path": str(material_path),
            "material": dict(material),
        })
        return True

    def set_environment(self, environment: dict[str, Any]) -> bool:
        self._raise_if_exited()
        self._queue_update({"type": "environment", "environment": dict(environment)})
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._sender_started:
            try:
                self._updates.put_nowait(None)
            except queue.Full:
                try:
                    self._updates.get_nowait()
                    self._updates.put_nowait(None)
                except queue.Empty:
                    pass
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"type":"close"}\n')
                self._process.stdin.flush()
                self._process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        try:
            self._protocol_listener.close()
        except OSError:
            pass
        if self._protocol_socket is not None:
            try:
                self._protocol_socket.close()
            except OSError:
                pass
        self._http_server.shutdown()
        self._http_server.server_close()


def _factory(session: Any, model: Any, config: dict[str, Any]) -> OVRTViewer:
    return OVRTViewer(session, model, config)


register_viewer("ovrtx", _factory)

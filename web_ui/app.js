const WS_PORT = 8001;
const ws = new WebSocket(`ws://localhost:${WS_PORT}`);

const floorCanvas = document.getElementById('floor-canvas');
const entitiesLayer = document.getElementById('entities-layer');
const svgLayer = document.getElementById('svg-layer');

let projectionCalibration = null;

// Generate unique DOM elements per footprint or text
const activeDOMEls = new Map(); 

// Listen for 'g' to toggle debug grid
document.addEventListener('keydown', (e) => {
    if (e.key === 'g' || e.key === 'G') {
        floorCanvas.classList.toggle('debug-grid');
    }
});

ws.onopen = () => {
    console.log('Connected to Projection Mapping Websocket');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    if (msg.type === 'calibration') {
        projectionCalibration = msg.data;
        applyHomographyTransform();
    } else if (msg.type === 'frame') {
        renderFrame(msg.person_trails, msg.matched_trails);
    }
};

function applyHomographyTransform() {
    if (!projectionCalibration) return;
    
    const H = projectionCalibration.H_proj;
    const h00 = H[0][0], h01 = H[0][1], h02 = H[0][2];
    const h10 = H[1][0], h11 = H[1][1], h12 = H[1][2];
    const h20 = H[2][0], h21 = H[2][1], h22 = H[2][2];
    
    const w = projectionCalibration.floor_w;
    const h = projectionCalibration.floor_h;

    floorCanvas.style.width = w + 'px';
    floorCanvas.style.height = h + 'px';

    floorCanvas.style.transform = `matrix3d(
        ${h00}, ${h10}, 0, ${h20},
        ${h01}, ${h11}, 0, ${h21},
        0, 0, 1, 0,
        ${h02}, ${h12}, 0, ${h22}
    )`;
}

function renderFrame(personTrails, matchedTrails) {
    let svgContents = '';
    const nowKeys = new Set();
    
    // For rendering, we will iterate over live matched paths
    Object.entries(matchedTrails).forEach(([pid, data]) => {
        const trail = data.trail;
        if (trail.length < 2) return;
        
        let dStr = getSplinePath(trail);
        const pathId = `path-${pid}`;
        
        svgContents += `
            <path id="${pathId}" class="fade-path" d="${dStr}" 
                  stroke="white" fill="transparent" stroke-width="8" />
        `;
        
        // Exact distance-based footprint placement
        const STEP_DISTANCE = 85; 
        
        let stepCount = 0;
        let distSinceLastStep = 0.0;
        
        // Always place a step exactly at the start
        const startPt = trail[0];
        if (trail.length > 1) {
             const dx = trail[1].x - startPt.x;
             const dy = trail[1].y - startPt.y;
             const rot = Math.atan2(dy, dx) * 180 / Math.PI;
             let imgId = `img-match-${pid}-${stepCount}`;
             nowKeys.add(imgId);
             updateOrCreateImg(imgId, startPt.x, startPt.y, rot, data.fade, '../white_foot.gif');
             stepCount++;
        }

        for (let i = 0; i < trail.length - 1; i++) {
            const p1 = trail[i];
            const p2 = trail[i+1];
            
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const segmentDist = Math.sqrt(dx*dx + dy*dy);
            
            if (segmentDist === 0) continue;
            
            const dirX = dx / segmentDist;
            const dirY = dy / segmentDist;
            const rot = Math.atan2(dirY, dirX) * 180 / Math.PI;
            
            let walkedOnSegment = 0.0;
            
            while (distSinceLastStep + (segmentDist - walkedOnSegment) >= STEP_DISTANCE) {
                const stepToTake = STEP_DISTANCE - distSinceLastStep;
                walkedOnSegment += stepToTake;
                distSinceLastStep = 0.0; // Reset since we stepped
                
                const stepX = p1.x + dirX * walkedOnSegment;
                const stepY = p1.y + dirY * walkedOnSegment;
                
                let imgId = `img-match-${pid}-${stepCount}`;
                nowKeys.add(imgId);
                // Drop footprint
                updateOrCreateImg(imgId, stepX, stepY, rot, data.fade, '../white_foot.gif');
                stepCount++;
            }
            distSinceLastStep += (segmentDist - walkedOnSegment);
        }
    });

    svgLayer.innerHTML = svgContents;
    
    // Unmount stale imgs
    activeDOMEls.forEach((el, id) => {
        if (!nowKeys.has(id)) {
            el.remove();
            activeDOMEls.delete(id);
        }
    });
}

function updateOrCreateImg(id, x, y, rot, opacity, srcUrl) {
    let img = activeDOMEls.get(id);
    if (!img) {
        img = document.createElement('img');
        img.id = id;
        img.src = srcUrl;
        img.style.position = 'absolute';
        
        img.style.width = '80px';
        img.style.height = '160px';
        
        img.style.transformOrigin = '50% 50%';
        img.style.transition = 'transform 0.05s linear, opacity 0.3s ease-out';
        
        entitiesLayer.appendChild(img);
        activeDOMEls.set(id, img);
    }
    
    img.style.opacity = Math.max(0, opacity);
    
    // Ensure we don't drop invalid numbers into CSS, which causes Fallback to 0,0 location!
    if (isNaN(x) || isNaN(y) || isNaN(rot)) {
        console.warn("Invalid footprint coords:", id, x, y, rot);
        return;
    }
    img.style.transform = `translate3d(${x - Math.round(80/2)}px, ${y - Math.round(160/2)}px, 0) rotate(${rot + 90}deg)`;
}

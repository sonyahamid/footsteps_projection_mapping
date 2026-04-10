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
        // Start playback loop once we have our calibration / transform
        if (!window.playbackStarted) {
            window.playbackStarted = true;
            startPlaybackLoop();
        }
    } 
    // We ignore 'frame' messages for live walkers now since we do async playback
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

const STEP_DISTANCE = 85; 

async function startPlaybackLoop() {
    while (true) {
        try {
            const res = await fetch('/api/random_path');
            if (res.ok) {
                const trailCoords = await res.json();
                if (trailCoords && Array.isArray(trailCoords) && trailCoords.length > 2) {
                    await playTrail(trailCoords);
                } else {
                    // No valid path data found, wait and try again
                    await new Promise(r => setTimeout(r, 2000));
                }
            }
        } catch (err) {
            console.error('Error fetching random path:', err);
        }
        
        // Random wait between 2s and 6s before playing the next trail
        const waitTime = 2000 + Math.random() * 4000;
        await new Promise(r => setTimeout(r, waitTime));
    }
}

async function playTrail(trail) {
    let dStr = getSplinePath(trail);
    const pid = Math.floor(Math.random() * 1000000); // Random ID for the instance
    const pathId = `path-${pid}`;
    
    // Draw the debug path
    svgLayer.innerHTML = `
        <path id="${pathId}" class="fade-path" d="${dStr}" 
              stroke="white" fill="transparent" stroke-width="8" opacity="0.3" />
    `;
    
    let steps = [];
    const startPt = trail[0];
    
    if (trail.length > 1) {
        const dx = trail[1].x - startPt.x;
        const dy = trail[1].y - startPt.y;
        const rot = Math.atan2(dy, dx) * 180 / Math.PI;
        steps.push({ x: startPt.x, y: startPt.y, rot: rot });
    }

    let distSinceLastStep = 0.0;
    
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
            distSinceLastStep = 0.0;
            
            const stepX = p1.x + dirX * walkedOnSegment;
            const stepY = p1.y + dirY * walkedOnSegment;
            
            steps.push({ x: stepX, y: stepY, rot: rot });
        }
        distSinceLastStep += (segmentDist - walkedOnSegment);
    }
    
    const stepIds = [];
    
    // Fade in chronologically
    for (let i = 0; i < steps.length; i++) {
        const st = steps[i];
        const imgId = `img-play-${pid}-${i}`;
        stepIds.push(imgId);
        
        // Start as transparent
        updateOrCreateImg(imgId, st.x, st.y, st.rot, 0, '../white_foot.gif');
        
        // Let the DOM update
        await new Promise(r => requestAnimationFrame(r));
        
        // Fade in
        updateOrCreateImg(imgId, st.x, st.y, st.rot, 1.0, '../white_foot.gif');
        
        // Walk pace delay (e.g. 300ms per step)
        await new Promise(r => setTimeout(r, 400));
    }
    
    // Pause briefly once full path is revealed
    await new Promise(r => setTimeout(r, 3000));
    
    // Fade out chronologically
    for (let i = 0; i < steps.length; i++) {
        const id = stepIds[i];
        const st = steps[i];
        
        // Fade out
        updateOrCreateImg(id, st.x, st.y, st.rot, 0, '../white_foot.gif');
        
        // Small delay between fading consecutive steps out
        await new Promise(r => setTimeout(r, 300));
    }
    
    // Remove all after fade outs finish
    await new Promise(r => setTimeout(r, 500)); // allow CSS transition
    stepIds.forEach(id => {
        const el = activeDOMEls.get(id);
        if (el) {
            el.remove();
            activeDOMEls.delete(id);
        }
    });

    svgLayer.innerHTML = ''; // clear debugging path
}

// Keep the old updateOrCreateImg function the same
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

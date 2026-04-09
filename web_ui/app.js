const WS_PORT = 8001;
const ws = new WebSocket(`ws://localhost:${WS_PORT}`);

const floorCanvas = document.getElementById('floor-canvas');
const entitiesLayer = document.getElementById('entities-layer');
const svgLayer = document.getElementById('svg-layer');

let projectionCalibration = null;

// Generate unique DOM elements per footprint or text
const activeDOMEls = new Map(); 

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
    
    // Homography from Floor -> Projector is H_proj in calibration.json
    const H = projectionCalibration.H_proj;
    
    // H is a 3x3 matrix. CSS matrix3d takes a 4x4 matrix (column-major order).
    // Let's expand:
    // [ h00, h01, h02 ]
    // [ h10, h11, h12 ]
    // [ h20, h21, h22 ]
    
    const h00 = H[0][0], h01 = H[0][1], h02 = H[0][2];
    const h10 = H[1][0], h11 = H[1][1], h12 = H[1][2];
    const h20 = H[2][0], h21 = H[2][1], h22 = H[2][2];
    
    const w = projectionCalibration.floor_w;
    const h = projectionCalibration.floor_h;

    // Apply the 3D transform mapping onto the main root div
    floorCanvas.style.width = w + 'px';
    floorCanvas.style.height = h + 'px';

    // We do NOT normalize by h22 if we want proper perspective projection.
    // CSS matrix3d:
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
    // Render Matched historical footprints and Text On Path
    Object.entries(matchedTrails).forEach(([pid, data]) => {
        const trail = data.trail;
        if (trail.length < 2) return;
        
        let dStr = `M ${trail[0].x} ${trail[0].y} `;
        for (let i = 1; i < trail.length; i++) {
            dStr += `L ${trail[i].x} ${trail[i].y} `;
        }
        
        const pathId = `path-${pid}`;
        const strokeColor = `rgba(0, 255, 0, ${data.fade})`; // Faded green lines
        
        svgContents += `
            <path id="${pathId}" class="fade-path" d="${dStr}" 
                  stroke="${strokeColor}" fill="transparent" stroke-width="3" stroke-dasharray="10 10"/>
            <text fill="rgba(255,255,255,${data.fade})" font-size="28" font-family="'Courier New', Courier, monospace" font-weight="bold">
                <textPath href="#${pathId}" startOffset="50%" text-anchor="middle">${data.age_str}</textPath>
            </text>
        `;
        
        // Spawn footprint GIF at head!
        const px = trail[trail.length-1].x;
        const py = trail[trail.length-1].y;
        
        // basic angle
        const dx = trail[trail.length-1].x - trail[trail.length-2].x;
        const dy = trail[trail.length-1].y - trail[trail.length-2].y;
        const rot = Math.atan2(dy, dx) * 180 / Math.PI;

        const imgId = `img-match-${pid}`;
        nowKeys.add(imgId);
        updateOrCreateImg(imgId, px, py, rot + 90, data.fade, '../white_foot.gif'); 
    });

    // Render simple live positions
    Object.entries(personTrails).forEach(([pid, trail]) => {
        if (trail.length < 2) return;
        
        const px = trail[trail.length-1].x;
        const py = trail[trail.length-1].y;
        
        const dx = trail[trail.length-1].x - trail[trail.length-2].x;
        const dy = trail[trail.length-1].y - trail[trail.length-2].y;
        const rot = Math.atan2(dy, dx) * 180 / Math.PI;

        const imgId = `img-live-${pid}`;
        nowKeys.add(imgId);
        updateOrCreateImg(imgId, px, py, rot + 90, 1.0, '../white_foot3.gif'); 
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
        
        // Size it appropriately (the gif is around 27x54 according to python mapper)
        img.style.width = '27px';
        img.style.height = '54px';
        
        // Move registration/transform-origin to center of foot
        img.style.transformOrigin = '50% 50%';
        img.style.transition = 'transform 0.05s linear, opacity 0.3s ease-out';
        
        entitiesLayer.appendChild(img);
        activeDOMEls.set(id, img);
    }
    
    // Update transform
    img.style.opacity = Math.max(0, opacity);
    
    // Hardware accelerated GPU composite for translate rotation
    // Note: centering by subtracting half width/height
    img.style.transform = `translate3d(${x - 13.5}px, ${y - 27}px, 0) rotate(${rot}deg)`;
}

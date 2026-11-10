(() => {
  'use strict';

  const CHANNELS = 16;
  const BACKGROUND = [27 / 255, 27 / 255, 31 / 255];
  const canvas = document.querySelector('#simulation');
  const context = canvas.getContext('2d', { alpha: false });
  const image = context.createImageData(canvas.width, canvas.height);
  const playButton = document.querySelector('#play-button');
  const resetButton = document.querySelector('#reset-button');
  const modelInput = document.querySelector('#model-file');
  const speedInput = document.querySelector('#speed');
  const radiusInput = document.querySelector('#brush-radius');
  const colorInput = document.querySelector('#paint-color');
  const randomColorInput = document.querySelector('#random-color');
  const toolButtons = [...document.querySelectorAll('.tool-button')];
  let model = null;
  let cells = null;
  let frame = 0;
  let running = false;
  let lastTime = 0;
  let accumulator = 0;
  let activePointer = null;

  const selectedTool = () => document.querySelector('.tool-button.is-selected')?.dataset.tool || 'paint';

  function setStatus(label, isRunning = false) {
    const status = document.querySelector('#run-status');
    if (!status) return;
    status.lastChild.textContent = ` ${label}`;
    status.classList.toggle('is-running', isRunning);
  }

  function fail(message) {
    setStatus('ERROR');
    playButton.disabled = true;
    resetButton.disabled = true;
    const modelName = document.querySelector('#model-name');
    const canvasHint = document.querySelector('#canvas-hint');
    if (modelName) modelName.textContent = 'Model unavailable';
    if (canvasHint) canvasHint.textContent = message;
  }

  function validateModel(candidate) {
    if (!candidate || candidate.format !== 'nca-web-presentation' ||
        candidate.version !== 1 || candidate.channels !== CHANNELS ||
        !Number.isInteger(candidate.hidden) || candidate.hidden < 1 ||
        !Number.isFinite(candidate.fireRate) || candidate.fireRate < 0 ||
        candidate.fireRate > 1) {
      throw new Error('This JSON does not contain a supported NCA model.');
    }
    const hidden = candidate.hidden;
    const lengths = [
      ['fc1Weight', hidden * CHANNELS * 3],
      ['fc1Bias', hidden],
      ['fc2Weight', CHANNELS * hidden],
    ];
    for (const [key, expected] of lengths) {
      if (!Array.isArray(candidate[key]) || candidate[key].length !== expected ||
          candidate[key].some(value => !Number.isFinite(value))) {
        throw new Error(`Invalid model weights: ${key}.`);
      }
    }
    if (candidate.target &&
        (!Number.isInteger(candidate.target.width) ||
         !Number.isInteger(candidate.target.height) ||
         !Array.isArray(candidate.target.rgb) ||
         candidate.target.rgb.length !== candidate.target.width * candidate.target.height * 3)) {
      throw new Error('The optional target image in this JSON is invalid.');
    }
    return {
      ...candidate,
      fc1Weight: Float32Array.from(candidate.fc1Weight),
      fc1Bias: Float32Array.from(candidate.fc1Bias),
      fc2Weight: Float32Array.from(candidate.fc2Weight),
    };
  }

  function seed() {
    const width = canvas.width;
    const height = canvas.height;
    cells = new Float32Array(CHANNELS * width * height);
    const center = (Math.floor(height / 2) * width + Math.floor(width / 2)) * CHANNELS;
    for (let channel = 3; channel < CHANNELS; channel += 1) {
      cells[center + channel] = 1;
      if (model.config?.seed_cells !== 1 && Math.floor(width / 2) + 1 < width) {
        cells[center + CHANNELS + channel] = 1;
      }
    }
    frame = 0;
    render();
  }

  function isAlive(source, x, y) {
    for (let oy = -1; oy <= 1; oy += 1) {
      for (let ox = -1; ox <= 1; ox += 1) {
        const px = x + ox;
        const py = y + oy;
        if (px >= 0 && px < canvas.width && py >= 0 && py < canvas.height &&
            source[(py * canvas.width + px) * CHANNELS + 3] > 0.1) {
          return true;
        }
      }
    }
    return false;
  }

  function step() {
    const width = canvas.width;
    const height = canvas.height;
    const count = width * height;
    const perception = new Float32Array(count * CHANNELS * 3);
    const hidden = new Float32Array(count * model.hidden);
    const next = new Float32Array(cells.length);
    const kernelX = [-1 / 8, 0, 1 / 8, -2 / 8, 0, 2 / 8, -1 / 8, 0, 1 / 8];
    const kernelY = [-1 / 8, -2 / 8, -1 / 8, 0, 0, 0, 1 / 8, 2 / 8, 1 / 8];

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixel = y * width + x;
        const out = pixel * CHANNELS * 3;
        for (let channel = 0; channel < CHANNELS; channel += 1) {
          const base = out + channel * 3;
          perception[base] = cells[pixel * CHANNELS + channel];
          for (let ky = -1; ky <= 1; ky += 1) {
            for (let kx = -1; kx <= 1; kx += 1) {
              const px = x + kx;
              const py = y + ky;
              if (px < 0 || px >= width || py < 0 || py >= height) continue;
              const value = cells[(py * width + px) * CHANNELS + channel];
              const kernelIndex = (ky + 1) * 3 + kx + 1;
              perception[base + 1] += value * kernelX[kernelIndex];
              perception[base + 2] += value * kernelY[kernelIndex];
            }
          }
        }
      }
    }

    const inputSize = CHANNELS * 3;
    for (let pixel = 0; pixel < count; pixel += 1) {
      const pOffset = pixel * inputSize;
      const hOffset = pixel * model.hidden;
      for (let h = 0; h < model.hidden; h += 1) {
        let value = model.fc1Bias[h];
        const wOffset = h * inputSize;
        for (let i = 0; i < inputSize; i += 1) {
          value += model.fc1Weight[wOffset + i] * perception[pOffset + i];
        }
        hidden[hOffset + h] = Math.max(0, value);
      }
    }

    for (let pixel = 0; pixel < count; pixel += 1) {
      const hOffset = pixel * model.hidden;
      const offset = pixel * CHANNELS;
      const fired = Math.random() < model.fireRate;
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        let delta = 0;
        const wOffset = channel * model.hidden;
        for (let h = 0; h < model.hidden; h += 1) {
          delta += model.fc2Weight[wOffset + h] * hidden[hOffset + h];
        }
        next[offset + channel] = cells[offset + channel] + (fired ? delta : 0);
      }
    }

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixel = y * width + x;
        if (!isAlive(cells, x, y) || !isAlive(next, x, y)) {
          next.fill(0, pixel * CHANNELS, (pixel + 1) * CHANNELS);
        }
      }
    }
    cells = next;
    frame += 1;
  }

  function render() {
    for (let pixel = 0; pixel < canvas.width * canvas.height; pixel += 1) {
      const offset = pixel * CHANNELS;
      const alpha = Math.min(1, Math.max(0, cells[offset + 3]));
      const out = pixel * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        image.data[out + channel] = Math.round(
          Math.min(1, Math.max(0, cells[offset + channel] + (1 - alpha) * BACKGROUND[channel])) * 255
        );
      }
      image.data[out + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    const stepCount = document.querySelector('#step-count');
    if (stepCount) stepCount.textContent = `T+ ${String(frame).padStart(4, '0')}`;
  }

  function animate(timestamp) {
    if (!running) return;
    if (lastTime === 0) lastTime = timestamp;
    accumulator += Math.min(timestamp - lastTime, 100);
    lastTime = timestamp;
    const interval = 1000 / (Number(speedInput.value) * 15);
    let updated = false;
    while (accumulator >= interval) {
      step();
      accumulator -= interval;
      updated = true;
    }
    if (updated) render();
    requestAnimationFrame(animate);
  }

  function togglePlayback() {
    running = !running;
    document.querySelector('#play-icon').textContent = running ? 'Ⅱ' : '▶';
    document.querySelector('#play-label').textContent = running ? 'Pause' : 'Resume';
    setStatus(running ? 'RUNNING' : 'PAUSED', running);
    if (running) {
      lastTime = 0;
      requestAnimationFrame(animate);
    }
  }

  function loadModel(candidate) {
    model = validateModel(candidate);
    running = false;
    document.querySelector('#play-icon').textContent = '▶';
    document.querySelector('#play-label').textContent = 'Start';
    const modelName = document.querySelector('#model-name');
    const modelIteration = document.querySelector('#model-iteration');
    const modelArchitecture = document.querySelector('#model-architecture');
    if (modelName) modelName.textContent = model.name || 'NCA model';
    if (modelIteration) {
      modelIteration.textContent = Number(model.iteration || 0).toLocaleString('en-US');
    }
    if (modelArchitecture) modelArchitecture.textContent = `16 → ${model.hidden} → 16`;
    playButton.disabled = false;
    resetButton.disabled = false;
    const canvasHint = document.querySelector('#canvas-hint');
    if (canvasHint) canvasHint.textContent = 'Touch or click to add a new seed';
    setStatus('READY');
    seed();
  }

  playButton.addEventListener('click', togglePlayback);
  resetButton.addEventListener('click', () => {
    running = false;
    document.querySelector('#play-icon').textContent = '▶';
    document.querySelector('#play-label').textContent = 'Start';
    setStatus('READY');
    seed();
  });
  radiusInput.addEventListener('input', () => {
    document.querySelector('#radius-value').textContent = radiusInput.value;
  });
  speedInput.addEventListener('input', () => {
    document.querySelector('#speed-value').textContent = `${speedInput.value}×`;
  });
  function applyTool(event) {
    if (!model || !cells) return;
    const bounds = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(canvas.width - 1,
      Math.floor((event.clientX - bounds.left) / bounds.width * canvas.width)));
    const y = Math.max(0, Math.min(canvas.height - 1,
      Math.floor((event.clientY - bounds.top) / bounds.height * canvas.height)));
    const tool = selectedTool();

    if (tool === 'seed') {
      const seedCells = model.config?.seed_cells === 1 ? 1 : 2;
      for (let dx = 0; dx < seedCells && x + dx < canvas.width; dx += 1) {
        const offset = (y * canvas.width + x + dx) * CHANNELS;
        cells.fill(0, offset, offset + CHANNELS);
        for (let channel = 3; channel < CHANNELS; channel += 1) cells[offset + channel] = 1;
      }
    } else {
      const radius = Math.max(1, Number(radiusInput.value));
      const color = randomColorInput.checked
        ? Array.from({ length: 3 }, () => Math.random())
        : [1, 3, 5].map(index => parseInt(colorInput.value.slice(index, index + 2), 16) / 255);
      for (let py = Math.max(0, y - radius); py <= Math.min(canvas.height - 1, y + radius); py += 1) {
        for (let px = Math.max(0, x - radius); px <= Math.min(canvas.width - 1, x + radius); px += 1) {
          if ((px - x) ** 2 + (py - y) ** 2 > radius ** 2) continue;
          const offset = (py * canvas.width + px) * CHANNELS;
          if (tool === 'erase') {
            cells.fill(0, offset, offset + CHANNELS);
          } else {
            cells[offset] = color[0];
            cells[offset + 1] = color[1];
            cells[offset + 2] = color[2];
            cells[offset + 3] = 1;
            for (let channel = 4; channel < CHANNELS; channel += 1) {
              cells[offset + channel] = Math.random() * 2 - 1;
            }
          }
        }
      }
    }
    render();
  }

  toolButtons.forEach(button => button.addEventListener('click', () => {
    for (const candidate of toolButtons) {
      const selected = candidate === button;
      candidate.classList.toggle('is-selected', selected);
      candidate.setAttribute('aria-pressed', String(selected));
    }
  }));
  canvas.addEventListener('pointerdown', event => {
    if (!model) return;
    event.preventDefault();
    activePointer = event.pointerId;
    canvas.setPointerCapture(event.pointerId);
    applyTool(event);
  });
  canvas.addEventListener('pointermove', event => {
    if (event.pointerId === activePointer && selectedTool() !== 'seed') applyTool(event);
  });
  function endPointer(event) {
    if (event.pointerId === activePointer) activePointer = null;
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('lostpointercapture', endPointer);
  modelInput.addEventListener('change', async () => {
    const file = modelInput.files?.[0];
    if (!file) return;
    try {
      loadModel(JSON.parse(await file.text()));
    } catch (error) {
      fail(error.message || 'Die JSON-Datei konnte nicht geladen werden.');
    }
    modelInput.value = '';
  });

  fetch('model.json')
    .then(response => {
      if (!response.ok) throw new Error('model.json nicht gefunden');
      return response.json();
    })
    .then(loadModel)
    .catch(() => {
      setStatus('MODEL MISSING');
      const modelName = document.querySelector('#model-name');
      const canvasHint = document.querySelector('#canvas-hint');
      if (modelName) modelName.textContent = 'Load a model JSON';
      if (canvasHint) {
        canvasHint.textContent = 'Choose a model JSON or open this page from a static web server.';
      }
    });
})();

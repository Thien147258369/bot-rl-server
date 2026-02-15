/*
Bot RL Server (Node.js)

- WebSocket server for game clients to connect and control/train bots.
- Q-learning (tabular) implementation with simple discretization.
- Messages supported (JSON):
  1) { type: 'obs', botId: <id>, obs: { hp, ammo, dist, inZone, hasKnife } }
     -> Server replies: { type: 'action', botId: <id>, action: <actionString> }
  2) { type: 'reward', botId: <id>, reward: <number>, nextObs: { ... } }
     -> Server updates Q-table
  3) { type: 'save' } -> Server saves Q-table to disk

- HTTP endpoints (for quick inspection):
  - GET /qtable   -> returns small summary (may be large)
  - GET /save     -> forces a save

Run instructions:
  1) Save this file as bot-server.js on your server/machine.
  2) Install dependencies: npm init -y && npm i ws
  3) Run: node bot-server.js
  4) Connect your game client WebSocket to ws://<server-ip>:8080

Notes:
- Tune hyperparameters ALPHA/GAMMA/EPSILON below, and discretization buckets.
- Q-table stored in qtable.json in same folder.
- This server is intentionally simple and synchronous (file I/O is minimal).

*/

const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Configuration (changeable)
const WS_PORT = process.env.WS_PORT ? parseInt(process.env.WS_PORT) : 8080;
const HTTP_PORT = process.env.HTTP_PORT ? parseInt(process.env.HTTP_PORT) : 8080; // same port used for both (upgrade handled)
const SAVE_PATH = path.join(__dirname, 'qtable.json');
const SAVE_INTERVAL_MS = 30_000; // autosave interval

// RL hyperparameters
const ALPHA = parseFloat(process.env.ALPHA) || 0.12; // learning rate
const GAMMA = parseFloat(process.env.GAMMA) || 0.96;  // discount factor
let EPSILON = parseFloat(process.env.EPSILON) || 0.25; // exploration

// Actions: keep concise strings for client to handle
const ACTIONS = [
  'MOVE_TO_PLAYER',
  'MOVE_RANDOM',
  'SHOOT',
  'PICK_CRATE',
  'RETREAT',
  'IDLE'
];

// Internal Q-table: { stateKey: [q0,q1,q2,...] }
let qtable = {};

// Load qtable if exists
try {
  if (fs.existsSync(SAVE_PATH)) {
    const raw = fs.readFileSync(SAVE_PATH, 'utf8');
    qtable = JSON.parse(raw) || {};
    console.log('[Server] Loaded qtable from', SAVE_PATH);
  }
} catch (e) {
  console.warn('[Server] Failed to load qtable:', e.message);
}

// Utility: discretize observation into a small state key
function discretize(obs) {
  // Expect obs to include hp (0-100), ammo (0+), dist (distance in blocks), inZone (0/1), hasKnife (0/1)
  const hp = Math.max(0, Math.min(100, Math.round(obs.hp || 0)));
  const hpBucket = Math.floor(hp / 25); // 0..4
  const ammoBucket = (obs.ammo && obs.ammo > 0) ? 1 : 0; // 0/1
  const dist = Math.max(0, Math.round(obs.dist || 999));
  const distBucket = Math.min(4, Math.floor(dist / 20)); // 0..4
  const zone = obs.inZone ? 1 : 0;
  const knife = obs.hasKnife ? 1 : 0;
  return `${hpBucket}|${ammoBucket}|${distBucket}|${zone}|${knife}`;
}

function ensureState(key) {
  if (!qtable[key]) qtable[key] = Array(ACTIONS.length).fill(0);
}

function chooseAction(key) {
  ensureState(key);
  if (Math.random() < EPSILON) {
    return Math.floor(Math.random() * ACTIONS.length);
  }
  const qvals = qtable[key];
  let bestIdx = 0;
  let bestV = -Infinity;
  for (let i = 0; i < qvals.length; i++) {
    if (qvals[i] > bestV) { bestV = qvals[i]; bestIdx = i; }
  }
  return bestIdx;
}

function updateQ(s, a, r, sNext) {
  ensureState(s); ensureState(sNext);
  const qsa = qtable[s][a];
  const maxNext = Math.max(...qtable[sNext]);
  qtable[s][a] = qsa + ALPHA * (r + GAMMA * maxNext - qsa);
}

// Simple HTTP server that also upgrades to WebSocket
const server = http.createServer((req, res) => {
  if (req.url === '/qtable') {
    // caution: could be large
    res.writeHead(200, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({ size: Object.keys(qtable).length }));
    return;
  }
  if (req.url === '/save') {
    try { fs.writeFileSync(SAVE_PATH, JSON.stringify(qtable)); } catch(e){}
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('saved');
    return;
  }
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('Bot RL Server: WebSocket endpoint at ws://<host>:' + WS_PORT);
});

const wss = new WebSocket.Server({ server });

// Per-socket memory: store last state/action per botId for updates
// Structure: ws._botState = { [botId]: { state, actionIdx, lastObsAt } }

wss.on('connection', (ws, req) => {
  ws._botState = ws._botState || {};
  ws.isAlive = true;

  ws.on('pong', () => ws.isAlive = true);

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch (e) { return; }

    if (msg.type === 'obs') {
      // message: { type:'obs', botId, obs }
      const botId = String(msg.botId || 'unknown');
      const obs = msg.obs || {};
      const stateKey = discretize(obs);
      const actionIdx = chooseAction(stateKey);
      // store last
      ws._botState[botId] = { state: stateKey, actionIdx: actionIdx, lastObsAt: Date.now() };
      // send action to client
      ws.send(JSON.stringify({ type: 'action', botId: botId, action: ACTIONS[actionIdx] }));

    } else if (msg.type === 'reward') {
      // message: { type:'reward', botId, reward, nextObs }
      const botId = String(msg.botId || 'unknown');
      const reward = Number(msg.reward || 0);
      const nextObs = msg.nextObs || {};
      const rec = ws._botState[botId];
      if (rec && typeof rec.actionIdx !== 'undefined') {
        const nextKey = discretize(nextObs);
        updateQ(rec.state, rec.actionIdx, reward, nextKey);
        // optionally remove stored entry
        delete ws._botState[botId];
      }
    } else if (msg.type === 'save') {
      try { fs.writeFileSync(SAVE_PATH, JSON.stringify(qtable)); }
      catch(e){}
      ws.send(JSON.stringify({ type: 'ok', msg: 'saved' }));
    } else if (msg.type === 'ping') {
      ws.send(JSON.stringify({ type: 'pong' }));
    }
  });

  ws.on('close', () => {
    ws._botState = null;
  });
});

// Periodic save and epsilon decay
setInterval(() => {
  try { fs.writeFileSync(SAVE_PATH, JSON.stringify(qtable)); } catch(e) { console.warn('save fail', e.message); }
  // decay epsilon slowly
  EPSILON = Math.max(0.01, EPSILON * 0.9995);
  console.log('[Server] autosaved qtable. states=', Object.keys(qtable).length, 'EPSILON=', EPSILON.toFixed(3));
}, SAVE_INTERVAL_MS);

// Ping clients to keep connections healthy
setInterval(() => {
  wss.clients.forEach((c) => {
    if (!c.isAlive) return c.terminate();
    c.isAlive = false;
    c.ping(() => {});
  });
}, 30_000);

server.listen(HTTP_PORT, () => {
  console.log('[Server] listening on', HTTP_PORT, ' (WebSocket upgrade supported)');
  console.log('[Server] Actions:', ACTIONS.join(', '));
});

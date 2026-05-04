import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import Globe from 'react-globe.gl';
import axios from 'axios';
import { Mic, Plus, Trash2, Activity, Info, X, Trees } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

const API_BASE = 'http://localhost:8000';

// Major forest regions for procedural tree placement
const FOREST_ZONES = [
  { lat: -3.46, lng: -62.2, spread: 15, density: 400 }, // Amazon
  { lat: -0.5, lng: 23.5, spread: 10, density: 250 },  // Congo
  { lat: 60.0, lng: 100.0, spread: 20, density: 300 }, // Taiga
  { lat: 45.0, lng: -120.0, spread: 8, density: 150 }, // Pacific Northwest
  { lat: -25.0, lng: 130.0, spread: 5, density: 50 },   // Australian Tropics
  { lat: 35.0, lng: 140.0, spread: 4, density: 80 },   // Japan
  { lat: 50.0, lng: 10.0, spread: 10, density: 200 },  // European Forests
];

interface Node {
  id: string;
  lat: number;
  lng: number;
  name: string;
  status: 'active' | 'alert' | 'warning';
}

interface Prediction {
  label: string;
  confidence: number;
}

const App: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [streamingNodes, setStreamingNodes] = useState<Set<string>>(new Set());
  const [nodePredictions, setNodePredictions] = useState<Record<string, Prediction[]>>({});
  const [availableSamples, setAvailableSamples] = useState<string[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [isAddingNode, setIsAddingNode] = useState(false);
  
  const globeRef = useRef<any>();
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const socketsRef = useRef<Map<string, WebSocket>>(new Map());
  const simulationsRef = useRef<Map<string, { interval: any, audioContext: AudioContext }>>(new Map());
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Connection web logic
  const nodeConnections = useMemo(() => {
    const connections: any[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];
        const dist = Math.sqrt(Math.pow(n1.lat - n2.lat, 2) + Math.pow(n1.lng - n2.lng, 2));
        if (dist < 20) {
          connections.push({
            startLat: n1.lat,
            startLng: n1.lng,
            endLat: n2.lat,
            endLng: n2.lng
          });
        }
      }
    }
    return connections;
  }, [nodes]);

  // Helper to encode raw PCM to WAV
  const encodeWAV = (samples: Float32Array, sampleRate: number) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 32 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  };

  // Generate 3D Trees data
  const trees = useMemo(() => {
    const data: any[] = [];
    FOREST_ZONES.forEach(zone => {
      for (let i = 0; i < zone.density; i++) {
        data.push({
          lat: zone.lat + (Math.random() - 0.5) * zone.spread,
          lng: zone.lng + (Math.random() - 0.5) * zone.spread,
          size: 0.2 + Math.random() * 0.4,
          type: Math.random() > 0.5 ? 'pine' : 'broad'
        });
      }
    });
    return data;
  }, []);

  useEffect(() => {
    fetchNodes();
    fetchSamples();
    setTimeout(() => {
      if (globeRef.current) {
        globeRef.current.pointOfView({ altitude: 1.5 }, 2000);
      }
    }, 100);
  }, []);

  useEffect(() => {
    if (selectedNode && globeRef.current) {
      globeRef.current.pointOfView({
        lat: selectedNode.lat,
        lng: selectedNode.lng,
        altitude: 0.15
      }, 2000);
    }
  }, [selectedNode]);

  const fetchNodes = async () => {
    try {
      const res = await axios.get(`${API_BASE}/nodes`);
      setNodes(res.data);
    } catch (err) {
      console.error("Failed to fetch nodes", err);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await axios.get(`${API_BASE}/available-samples`);
      setAvailableSamples(res.data);
      if (res.data.length > 0) setSelectedSample(res.data[0]);
    } catch (err) {
      console.error("Failed to fetch samples", err);
    }
  };

  const handleGlobeClick = useCallback(({ lat, lng }: { lat: number, lng: number }) => {
    if (isAddingNode) {
      const name = prompt("Enter sensor name:", `Forest Station ${nodes.length + 1}`);
      if (name) {
        const newNode: Node = {
          id: Math.random().toString(36).substr(2, 9),
          lat,
          lng,
          name,
          status: 'active'
        };
        addNode(newNode);
      }
      setIsAddingNode(false);
    }
  }, [isAddingNode, nodes]);

  const addNode = async (node: Node) => {
    try {
      await axios.post(`${API_BASE}/nodes`, node);
      setNodes(prev => [...prev, node]);
      setSelectedNode(node);
    } catch (err) {
      console.error("Failed to add node", err);
    }
  };

  const resolveAlert = (nodeId: string) => {
    setNodes(prev => prev.map(n => 
      n.id === nodeId ? { ...n, status: 'active' } : n
    ));
    if (selectedNode?.id === nodeId) {
      setSelectedNode({ ...selectedNode, status: 'active' });
    }
  };

  const deleteNode = async (id: string) => {
    if (!confirm("Are you sure you want to remove this station?")) return;
    try {
      stopSimulation(id);
      await axios.delete(`${API_BASE}/nodes/${id}`);
      setNodes(prev => prev.filter(n => n.id !== id));
      if (selectedNode?.id === id) setSelectedNode(null);
    } catch (err) {
      console.error("Failed to delete node", err);
    }
  };

  const startRecording = async () => {
    if (!selectedNode) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext({ sampleRate: 32000 });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      const samples: Float32Array[] = [];

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        samples.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);

      setTimeout(() => {
        if (audioContext.state !== 'closed') {
          // Stop recording
          source.disconnect();
          processor.disconnect();
          audioContext.close();
          setIsRecording(false);
          
          // Flatten samples
          const flattened = new Float32Array(samples.reduce((acc, curr) => acc + curr.length, 0));
          let offset = 0;
          for (const s of samples) {
            flattened.set(s, offset);
            offset += s.length;
          }

          const wavBlob = encodeWAV(flattened, 32000);
          sendAudio(wavBlob);
          stream.getTracks().forEach(track => track.stop());
        }
      }, 5000);

    } catch (err) {
      console.error("Microphone access denied", err);
    }
  };

  const sendAudio = async (blob: Blob) => {
    if (!selectedNode) return;
    const formData = new FormData();
    formData.append('file', blob, 'audio.wav');
    
    try {
      const res = await axios.post(`${API_BASE}/process-audio/${selectedNode.id}`, formData);
      setNodePredictions(prev => ({ ...prev, [selectedNode.id]: res.data.predictions }));
    } catch (err) {
      console.error("Failed to process audio", err);
    }
  };

  const startSimulation = async (nodeId: string, sampleFile: string) => {
    if (!nodeId || !sampleFile) return;
    
    try {
      // 1. Load and decode audio
      const response = await fetch(`${API_BASE}/samples/${sampleFile}`);
      const arrayBuffer = await response.arrayBuffer();
      
      const audioContext = new AudioContext({ sampleRate: 32000 });
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // 2. Setup WebSocket
      const ws = new WebSocket(`ws://localhost:8000/ws/stream/${nodeId}`);
      socketsRef.current.set(nodeId, ws);

      ws.onopen = () => {
        setStreamingNodes(prev => new Set(prev).add(nodeId));
        
        // 3. Start streaming loop
        let offset = 0;
        const chunkSize = 4096;
        const intervalTime = (chunkSize / 32000) * 1000;
        
        const interval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            const chunk = new Float32Array(chunkSize);
            const channelData = audioBuffer.getChannelData(0);
            
            for (let i = 0; i < chunkSize; i++) {
              chunk[i] = channelData[(offset + i) % audioBuffer.length];
            }
            
            ws.send(chunk);
            offset += chunkSize;
          }
        }, intervalTime);
        
        simulationsRef.current.set(nodeId, { interval, audioContext });
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.predictions) {
          setNodePredictions(prev => ({ ...prev, [data.node_id]: data.predictions }));
          
          const threats = ['chainsaw', 'axe', 'gunshot', 'truck', 'engine', 'fire', 'crackling_fire'];
          const isThreat = data.predictions.some((p: any) => 
            threats.includes(p.label.toLowerCase()) && p.confidence > 0.6
          );

          if (isThreat) {
            setNodes(prev => prev.map(n => 
              n.id === data.node_id ? { ...n, status: 'alert' } : n
            ));
            setSelectedNode(prev => (prev && prev.id === data.node_id) ? { ...prev, status: 'alert' } : prev);
          }
        }
      };

      ws.onclose = () => {
        stopSimulation(nodeId);
      };

    } catch (err) {
      console.error("Simulation setup failed", err);
    }
  };

  const stopSimulation = (nodeId: string) => {
    const sim = simulationsRef.current.get(nodeId);
    if (sim) {
      clearInterval(sim.interval);
      sim.audioContext.close();
      simulationsRef.current.delete(nodeId);
    }
    
    const ws = socketsRef.current.get(nodeId);
    if (ws) {
      ws.close();
      socketsRef.current.delete(nodeId);
    }
    
    setStreamingNodes(prev => {
      const next = new Set(prev);
      next.delete(nodeId);
      return next;
    });
  };

  const startStreaming = async () => {
    if (!selectedNode) return;
    startSimulation(selectedNode.id, selectedSample);
  };

  const stopStreaming = () => {
    if (!selectedNode) return;
    stopSimulation(selectedNode.id);
  };

  return (
    <div className="relative w-full h-screen bg-[#050505] overflow-hidden font-sans">
      {/* Globe Container */}
      <div className="absolute inset-0 cursor-crosshair">
        <Globe
          ref={globeRef}
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
          backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
          
          globeResolution={80}
          showAtmosphere={true}
          atmosphereColor="#22c55e"
          atmosphereAltitude={0.12}

          // Procedural Trees
          customLayerData={trees}
          customThreeObject={(d: any) => {
            const group = new THREE.Group();
            const trunk = new THREE.Mesh(
              new THREE.CylinderGeometry(0.015, 0.02, 0.15),
              new THREE.MeshLambertMaterial({ color: '#3d2b1f' })
            );
            trunk.position.y = 0.075;
            const leavesMaterial = new THREE.MeshLambertMaterial({ color: d.type === 'pine' ? '#143d14' : '#2d5a27' });
            const leaves = d.type === 'pine' 
              ? new THREE.Mesh(new THREE.ConeGeometry(0.08, 0.3, 6), leavesMaterial)
              : new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), leavesMaterial);
            leaves.position.y = d.type === 'pine' ? 0.22 : 0.18;
            leaves.scale.set(d.size, d.size, d.size);
            group.add(trunk, leaves);
            return group;
          }}
          customThreeObjectUpdate={(obj, d: any) => {
            Object.assign(obj.position, globeRef.current.getCoords(d.lat, d.lng, 0));
            obj.lookAt(new THREE.Vector3(0, 0, 0));
            obj.rotateX(Math.PI / 2);
          }}

          // Connection Web
          arcsData={nodeConnections}
          arcColor={() => 'rgba(34, 197, 94, 0.15)'}
          arcDashLength={0.5}
          arcDashGap={1}
          arcDashAnimateTime={3000}
          arcStroke={0.04}
          arcAltitudeAutoScale={0.2}

          // 3D Nodes (Realistic Microphone style)
          pointsData={nodes}
          pointLat="lat"
          pointLng="lng"
          pointColor={(d: any) => d.status === 'alert' ? "#ff0000" : "#333333"}
          pointAltitude={0.06}
          pointRadius={0.15}
          pointsMerge={false}

          // Pulsing Rings
          ringsData={nodes}
          ringLat="lat"
          ringLng="lng"
          ringColor={(d: any) => d.status === 'alert' ? "#ef4444" : (streamingNodes.has(d.id)) ? "#3b82f6" : "#22c55e"}
          ringMaxRadius={(d: any) => d.status === 'alert' ? 12 : (streamingNodes.has(d.id)) ? 6 : 2.5}
          ringPropagationSpeed={(d: any) => d.status === 'alert' ? 6 : 2}
          ringRepeatPeriod={(d: any) => d.status === 'alert' ? 250 : 800}

          // Labels
          labelsData={nodes}
          labelLat="lat"
          labelLng="lng"
          labelText={(d: any) => d.status === 'alert' ? `⚠ INTRUSION: ${d.name}` : (streamingNodes.has(d.id)) ? `● LIVE: ${d.name}` : d.name}
          labelSize={0.7}
          labelDotRadius={0.06}
          labelColor={(d: any) => d.status === 'alert' ? "#ff0000" : "white"}
          labelResolution={2}
          labelAltitude={0.08}

          onLabelClick={(d: any) => setSelectedNode(d)}
          onPointClick={(d: any) => setSelectedNode(d)}
          onGlobeClick={handleGlobeClick}
        />
      </div>

      {/* Overlay UI */}
      <div className="absolute top-0 left-0 w-full p-6 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto bg-black/60 backdrop-blur-2xl border border-white/10 p-6 rounded-[2.5rem] shadow-2xl">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-3 bg-green-500/20 rounded-2xl border border-green-500/30">
              <Trees className="text-green-400" size={28} />
            </div>
            <div>
              <h1 className="text-3xl font-black tracking-tighter text-white uppercase italic leading-none">
                Terra<span className="text-green-500">Guard</span>
              </h1>
              <p className="text-green-500/60 text-[10px] font-black uppercase tracking-[0.3em]">Neural Forest Watch</p>
            </div>
          </div>
        </div>

        <div className="pointer-events-auto flex flex-col gap-3 items-end">
          <button 
            onClick={() => setIsAddingNode(!isAddingNode)}
            className={`group flex items-center gap-3 px-8 py-4 rounded-full font-black transition-all ${
              isAddingNode ? 'bg-red-500 text-white' : 'bg-white text-black hover:bg-green-500 hover:text-white'
            } shadow-[0_20px_40px_rgba(0,0,0,0.4)] active:scale-95`}
          >
            {isAddingNode ? <X size={22} /> : <Plus size={22} />}
            <span className="uppercase tracking-widest">{isAddingNode ? 'Cancel' : 'Deploy Sensor'}</span>
          </button>
          
          <div className="bg-black/40 backdrop-blur-md px-6 py-4 rounded-[2rem] border border-white/5 text-right flex items-center gap-4">
            <div>
              <div className="text-[10px] text-white/40 uppercase font-black tracking-widest">Active Stations</div>
              <div className="text-2xl font-mono text-white leading-none">{nodes.length}</div>
            </div>
            <div className="w-10 h-10 rounded-full border-2 border-green-500/20 flex items-center justify-center">
               <Activity size={18} className="text-green-500 animate-pulse" />
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div 
            initial={{ x: 600 }}
            animate={{ x: 0 }}
            exit={{ x: 600 }}
            className="absolute right-0 top-0 h-full w-[500px] bg-black/90 backdrop-blur-3xl border-l border-white/10 p-12 overflow-y-auto"
          >
            <div className="flex justify-between items-start mb-16">
              <div className="relative">
                <div className="absolute -left-6 top-1/2 -translate-y-1/2 w-1 h-12 bg-green-500 rounded-full" />
                <span className="text-green-500 text-[10px] font-black uppercase tracking-[0.4em] mb-2 block">Station ID: {selectedNode.id.toUpperCase()}</span>
                <h2 className="text-4xl font-black text-white leading-tight uppercase italic">{selectedNode.name}</h2>
              </div>
              <button onClick={() => setSelectedNode(null)} className="p-3 bg-white/5 rounded-full hover:bg-white/10 transition-all active:scale-90">
                <X size={24} className="text-white/60" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-12">
              <div className="p-6 bg-white/5 rounded-[2rem] border border-white/10 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-30 transition-opacity">
                  <Activity size={40} className="text-green-500" />
                </div>
                <div className="text-[10px] text-white/40 uppercase font-black mb-1 tracking-widest">Latitude</div>
                <div className="text-xl font-mono text-white">{selectedNode.lat.toFixed(6)}°</div>
              </div>
              <div className="p-6 bg-white/5 rounded-[2rem] border border-white/10 relative overflow-hidden group">
                 <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-30 transition-opacity">
                  <Activity size={40} className="text-green-500" />
                </div>
                <div className="text-[10px] text-white/40 uppercase font-black mb-1 tracking-widest">Longitude</div>
                <div className="text-xl font-mono text-white">{selectedNode.lng.toFixed(6)}°</div>
              </div>
            </div>

            <div className="space-y-4 mb-16">
              {selectedNode.status === 'alert' && (
                <button
                  onClick={() => resolveAlert(selectedNode.id)}
                  className="w-full py-6 mb-4 bg-red-500 hover:bg-red-600 text-white rounded-[2rem] font-black uppercase tracking-widest flex items-center justify-center gap-4 animate-pulse shadow-[0_0_30px_rgba(239,68,68,0.4)]"
                >
                  <Info size={20} />
                  Resolve Alert
                </button>
              )}

              <div className="flex flex-col gap-2 mb-4">
                <label className="text-[10px] text-white/40 uppercase font-black tracking-widest ml-4">Simulation Source</label>
                <select 
                  value={selectedSample}
                  onChange={(e) => setSelectedSample(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-full px-6 py-4 text-white uppercase tracking-widest font-bold focus:outline-none focus:border-green-500/50 appearance-none"
                  disabled={streamingNodes.has(selectedNode.id)}
                >
                  {availableSamples.map(sample => (
                    <option key={sample} value={sample} className="bg-black text-white">{sample.replace(/\.[^/.]+$/, "").replace(/_/g, " ")}</option>
                  ))}
                </select>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={startRecording}
                  disabled={isRecording || streamingNodes.has(selectedNode.id)}
                  className={`flex-1 flex items-center justify-center gap-4 py-6 rounded-[2rem] transition-all font-black uppercase tracking-widest ${
                    isRecording ? 'bg-red-500 animate-pulse' : 'bg-white text-black hover:bg-green-500 hover:text-white disabled:opacity-50'
                  } shadow-2xl`}
                >
                  <Mic size={20} />
                  {isRecording ? 'Listening...' : 'Quick Scan'}
                </button>
                
                <button
                  onClick={streamingNodes.has(selectedNode.id) ? stopStreaming : startStreaming}
                  className={`flex-1 flex items-center justify-center gap-4 py-6 rounded-[2rem] transition-all font-black uppercase tracking-widest ${
                    streamingNodes.has(selectedNode.id) ? 'bg-red-600 animate-pulse' : 'bg-green-600 text-white hover:bg-green-500 disabled:opacity-50'
                  } shadow-2xl`}
                >
                  <Activity size={20} />
                  {streamingNodes.has(selectedNode.id) ? 'Live' : 'Stream Live'}
                </button>
              </div>
              
              <button 
                onClick={() => deleteNode(selectedNode.id)}
                className="w-full py-4 text-white/20 hover:text-red-500 transition-all text-[10px] font-black uppercase tracking-[0.3em]"
              >
                Terminate Connection
              </button>
            </div>

            {(nodePredictions[selectedNode.id] || []).length > 0 && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <div className="flex items-center gap-4 mb-8">
                  <span className="text-[11px] font-black text-green-500 uppercase tracking-[0.5em] whitespace-nowrap">Neural Inference</span>
                  <div className="h-[1px] w-full bg-white/10" />
                </div>

                <div className="space-y-4">
                  {(nodePredictions[selectedNode.id] || []).map((p, i) => (
                    <div key={i} className="p-8 bg-gradient-to-br from-white/10 to-transparent rounded-[2.5rem] border border-white/10 group hover:border-green-500/30 transition-colors">
                      <div className="flex justify-between items-end mb-4">
                        <div>
                          <div className="text-[10px] text-white/30 uppercase font-black mb-1">Classification</div>
                          <span className="capitalize text-white text-2xl font-black tracking-tight italic uppercase">{p.label.replace(/_/g, ' ')}</span>
                        </div>
                        <div className="text-3xl font-mono text-green-400 font-black">
                          {(p.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${p.confidence * 100}%` }}
                          transition={{ duration: streamingNodes.has(selectedNode.id) ? 0.3 : 1.5, ease: "easeOut" }}
                          className={`h-full ${streamingNodes.has(selectedNode.id) ? 'bg-red-500 shadow-[0_0_20px_#ef4444]' : 'bg-green-500 shadow-[0_0_20px_#22c55e]'}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute bottom-10 left-10 flex items-center gap-12">
        <div className="flex flex-col">
          <div className="text-[10px] text-white/20 uppercase font-black tracking-widest mb-1">Core Engine</div>
          <div className="flex items-center gap-2 text-white/60 text-[11px] font-black uppercase tracking-widest">
            <Activity size={14} className="text-green-500" />
            CRNN-FOREST-V4.2
          </div>
        </div>
        <div className="flex flex-col">
          <div className="text-[10px] text-white/20 uppercase font-black tracking-widest mb-1">Visualization</div>
          <div className="flex items-center gap-2 text-white/60 text-[11px] font-black uppercase tracking-widest">
            <Trees size={14} className="text-green-400" />
            3D-INSTANCED-TERRAIN
          </div>
        </div>
      </div>
    </div>
  );
};


export default App;

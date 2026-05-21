import React, { useState, useMemo } from 'react';
import { Trees, Activity, Wifi, WifiOff, LayoutDashboard } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import SensorMap from './components/SensorMap';
import AlertTimeline from './components/AlertTimeline';
import ConfidencePanel from './components/ConfidencePanel';
import SensorHealth from './components/SensorHealth';

const App: React.FC = () => {
  const { isConnected, lastEvent, sensorStates } = useWebSocket();
  const [selectedSensorId, setSelectedSensorId] = useState<string | null>(null);

  const selectedSensorData = useMemo(() => {
    if (!selectedSensorId) return null;
    return sensorStates[selectedSensorId] || null;
  }, [selectedSensorId, sensorStates]);

  const alertCount = useMemo(() => {
    return Object.values(sensorStates).filter(s => s.risk_level !== 'LOW').length;
  }, [sensorStates]);

  return (
    <div className="relative w-full h-screen bg-[#050505] flex flex-col font-sans text-white selection:bg-green-500/30">
      {/* Top Bar */}
      <header className="h-20 border-b border-white/5 px-8 flex items-center justify-between bg-black/40 backdrop-blur-xl z-50">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/20 rounded-xl border border-green-500/30 shadow-[0_0_15px_rgba(34,197,94,0.2)]">
              <Trees className="text-green-400" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tighter uppercase italic leading-none">
                Terra<span className="text-green-500">Guard</span>
              </h1>
              <p className="text-green-500/60 text-[8px] font-black uppercase tracking-[0.3em]">Neural Forest Watch</p>
            </div>
          </div>
          
          <div className="h-8 w-[1px] bg-white/5 mx-2" />
          
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="text-[9px] text-white/30 uppercase font-black tracking-widest">Network Status</span>
              <div className="flex items-center gap-2">
                {isConnected ? (
                  <>
                    <Wifi size={12} className="text-green-400" />
                    <span className="text-[10px] font-bold text-green-400 uppercase tracking-wider">WebSocket Active</span>
                  </>
                ) : (
                  <>
                    <WifiOff size={12} className="text-red-500 animate-pulse" />
                    <span className="text-[10px] font-bold text-red-500 uppercase tracking-wider">Reconnecting...</span>
                  </>
                )}
              </div>
            </div>
            
            <div className="flex flex-col">
              <span className="text-[9px] text-white/30 uppercase font-black tracking-widest">Threat Level</span>
              <div className="flex items-center gap-2">
                <Activity size={12} className={alertCount > 0 ? "text-amber-500 animate-pulse" : "text-green-400"} />
                <span className={`text-[10px] font-bold uppercase tracking-wider ${alertCount > 0 ? "text-amber-500" : "text-green-400"}`}>
                  {alertCount > 0 ? `${alertCount} ACTIVE ALERTS` : 'ENVIRONMENT STABLE'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
            <div className="px-4 py-2 bg-white/5 rounded-full border border-white/5 flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-ping" />
                <span className="text-[10px] font-black uppercase tracking-widest text-white/60">Live feed: Moldova Cluster</span>
            </div>
            <button className="p-3 bg-white text-black rounded-full hover:bg-green-500 hover:text-white transition-all active:scale-95 shadow-xl">
                <LayoutDashboard size={18} />
            </button>
        </div>
      </header>

      {/* Main Layout */}
      <main className="flex-1 flex overflow-hidden p-6 gap-6 lg:flex-row flex-col">
        {/* Left Column: Map */}
        <section className="flex-[0.6] min-h-[400px]">
          <SensorMap 
            sensorStates={sensorStates} 
            onSelectSensor={setSelectedSensorId} 
          />
        </section>

        {/* Right Column: Panels */}
        <section className="flex-[0.4] flex flex-col gap-6 overflow-y-auto pr-2 custom-scrollbar">
          <div className="grid grid-cols-1 gap-6">
            <div className="h-[400px]">
              <ConfidencePanel sensorData={selectedSensorData} />
            </div>
            <div className="h-[350px]">
              <SensorHealth sensorStates={sensorStates} />
            </div>
            <div className="h-[500px]">
              <AlertTimeline lastEvent={lastEvent} />
            </div>
          </div>
        </section>
      </main>

      {/* Bottom status */}
      <footer className="h-10 border-t border-white/5 px-8 flex items-center justify-between bg-black/40 text-[9px] text-white/20 uppercase font-black tracking-[0.2em]">
        <div className="flex items-center gap-8">
            <span>Core: CRNN-FOREST-V4.2.0</span>
            <span>Region: South East Europe (MD)</span>
        </div>
        <div className="flex items-center gap-4">
            <span>Uptime: 99.98%</span>
            <div className="flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-green-500" />
                <span>Encrypted P2P Link</span>
            </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

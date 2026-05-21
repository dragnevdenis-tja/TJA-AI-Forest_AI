import React, { useState, useEffect } from 'react';
import { Shield } from 'lucide-react';
import type { InferenceResult } from '../hooks/useWebSocket';

interface SensorHealthProps {
  sensorStates: Record<string, InferenceResult>;
}

const SensorHealth: React.FC<SensorHealthProps> = ({ sensorStates }) => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 10000);
    return () => clearInterval(timer);
  }, []);

  const getStatus = (timestamp: string) => {
    const diff = (currentTime.getTime() - new Date(timestamp).getTime()) / 1000; // in seconds
    if (diff > 300) return 'offline'; // 5 mins
    if (diff > 120) return 'warning'; // 2 mins
    return 'online';
  };

  const formatRelativeTime = (timestamp: string) => {
    const diff = Math.floor((currentTime.getTime() - new Date(timestamp).getTime()) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  return (
    <div className="flex flex-col h-full bg-white/5 rounded-[2rem] border border-white/10 p-6">
      <div className="flex items-center gap-3 mb-6">
        <Shield size={20} className="text-green-500" />
        <h2 className="text-xl font-black text-white uppercase italic">Sensor Health</h2>
      </div>

      <div className="overflow-hidden">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-white/5">
              <th className="pb-3 text-[10px] text-white/30 uppercase font-black tracking-widest">Station ID</th>
              <th className="pb-3 text-[10px] text-white/30 uppercase font-black tracking-widest">Region</th>
              <th className="pb-3 text-[10px] text-white/30 uppercase font-black tracking-widest">Last Seen</th>
              <th className="pb-3 text-[10px] text-white/30 uppercase font-black tracking-widest text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {Object.values(sensorStates).map((sensor) => {
              const status = getStatus(sensor.timestamp);
              return (
                <tr key={sensor.sensor_id} className="group">
                  <td className="py-4 text-xs font-black text-white uppercase tracking-wider group-hover:text-green-500 transition-colors">
                    {sensor.sensor_id}
                  </td>
                  <td className="py-4 text-[10px] text-white/40 uppercase font-bold">
                    {sensor.region}
                  </td>
                  <td className="py-4 text-[10px] text-white/60 font-mono">
                    {formatRelativeTime(sensor.timestamp)}
                  </td>
                  <td className="py-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <span className={`text-[9px] font-black uppercase ${
                        status === 'online' ? 'text-green-400' : 
                        status === 'warning' ? 'text-amber-400' : 'text-red-500'
                      }`}>
                        {status}
                      </span>
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        status === 'online' ? 'bg-green-400 shadow-[0_0_8px_#4ade80]' : 
                        status === 'warning' ? 'bg-amber-400 animate-pulse' : 'bg-red-500'
                      }`} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-auto pt-6 border-t border-white/5 flex justify-between text-[9px] text-white/20 uppercase font-black tracking-widest">
        <span>Active Nodes: {Object.values(sensorStates).filter(s => getStatus(s.timestamp) === 'online').length}</span>
        <span>Mesh Network: v4.2 Stable</span>
      </div>
    </div>
  );
};

export default SensorHealth;

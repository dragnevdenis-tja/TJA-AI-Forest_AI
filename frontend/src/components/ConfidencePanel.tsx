import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { BarChart3, Activity } from 'lucide-react';
import type { InferenceResult } from '../hooks/useWebSocket';

interface ConfidencePanelProps {
  sensorData: InferenceResult | null;
}

const ConfidencePanel: React.FC<ConfidencePanelProps> = ({ sensorData }) => {
  if (!sensorData) {
    return (
      <div className="flex flex-col h-full bg-white/5 rounded-[2rem] border border-white/10 p-12 items-center justify-center text-center">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-6">
          <BarChart3 size={32} className="text-white/10" />
        </div>
        <h3 className="text-xl font-black text-white/40 uppercase tracking-widest">Select a sensor</h3>
        <p className="text-white/20 text-xs font-bold uppercase mt-2">to view live neural inference</p>
      </div>
    );
  }

  const data = Object.entries(sensorData.audio_predictions || {}).map(([name, value]) => ({
    name: name.toUpperCase(),
    value: value * 100
  })).sort((a, b) => b.value - a.value);

  const getBarColor = (name: string) => {
    const threats = ['FIRE', 'CHAINSAW', 'GUNSHOT'];
    if (threats.includes(name)) return '#ef4444';
    return '#22c55e';
  };

  return (
    <div className="flex flex-col h-full bg-white/5 rounded-[2rem] border border-white/10 p-8 overflow-hidden relative group">
      <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-20 transition-opacity">
        <Activity size={80} className="text-green-500" />
      </div>

      <div className="mb-8">
        <span className="text-green-500 text-[10px] font-black uppercase tracking-[0.4em] mb-2 block">
          Live Audio Analysis: {sensorData.sensor_id}
        </span>
        <h2 className="text-2xl font-black text-white uppercase italic leading-none flex items-center gap-3">
          <BarChart3 size={24} className="text-green-500" />
          Neural Confidence
        </h2>
      </div>

      <div className="flex-1 min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 20, right: 40 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              type="number" 
              domain={[0, 100]} 
              hide 
            />
            <YAxis 
              dataKey="name" 
              type="category" 
              width={80}
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10, fontWeight: 'bold' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip 
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-black/90 border border-white/10 p-3 rounded-xl backdrop-blur-xl">
                      <p className="text-[10px] font-black text-white uppercase">{payload[0].payload.name}</p>
                      <p className="text-xl font-mono text-green-400">{(payload[0].value as number).toFixed(1)}%</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar 
              dataKey="value" 
              radius={[0, 10, 10, 0]} 
              barSize={32}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.name)} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 flex justify-between items-end">
        <div>
          <div className="text-[10px] text-white/30 uppercase font-black mb-1">Inference Engine</div>
          <div className="text-xs text-white/60 font-bold uppercase tracking-wider">CRNN-SOUND-V2.0</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] text-white/30 uppercase font-black mb-1">Latency</div>
          <div className="text-xs text-green-400 font-mono font-bold">124ms</div>
        </div>
      </div>
    </div>
  );
};

export default ConfidencePanel;

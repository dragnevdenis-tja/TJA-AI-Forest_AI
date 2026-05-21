import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, Clock, MapPin } from 'lucide-react';
import type { WSEvent } from '../hooks/useWebSocket';

interface AlertTimelineProps {
  lastEvent: WSEvent | null;
}

const AlertTimeline: React.FC<AlertTimelineProps> = ({ lastEvent }) => {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [filter, setFilter] = useState<'ALL' | 'HIGH' | 'MEDIUM'>('ALL');

  // Initial load
  useEffect(() => {
    // In a real app, we'd fetch for all sensors, but let's just start fresh or mock
    // fetchRiskHistory('all')...
  }, []);

  // Handle live events
  useEffect(() => {
    if (lastEvent?.event === 'alert') {
        setAlerts(prev => [{
            id: Date.now(),
            sensor_id: lastEvent.sensor_id,
            risk_level: lastEvent.risk_level,
            timestamp: lastEvent.timestamp,
            triggered_by: 'Audio Detection' // Generic for now
        }, ...prev].slice(0, 50));
    }
  }, [lastEvent]);

  const filteredAlerts = alerts.filter(a => filter === 'ALL' || a.risk_level === filter);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'HIGH': return 'bg-red-500 text-white';
      case 'MEDIUM': return 'bg-amber-500 text-black';
      default: return 'bg-green-500 text-white';
    }
  };

  return (
    <div className="flex flex-col h-full bg-white/5 rounded-[2rem] border border-white/10 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-black text-white uppercase italic flex items-center gap-2">
          <AlertCircle size={20} className="text-red-500" />
          Alert Timeline
        </h2>
        
        <div className="flex gap-2">
          {(['ALL', 'HIGH', 'MEDIUM'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded-full text-[10px] font-black uppercase transition-all ${
                filter === f ? 'bg-white text-black' : 'bg-white/10 text-white/40 hover:bg-white/20'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
        <AnimatePresence initial={false}>
          {filteredAlerts.length === 0 ? (
            <div className="h-full flex items-center justify-center text-white/20 text-xs font-bold uppercase tracking-widest">
              No active alerts
            </div>
          ) : (
            filteredAlerts.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="p-4 bg-white/5 rounded-2xl border border-white/5 hover:border-white/20 transition-colors group"
              >
                <div className="flex justify-between items-start mb-2">
                  <span className={`px-2 py-0.5 rounded text-[8px] font-black uppercase ${getRiskColor(alert.risk_level)}`}>
                    {alert.risk_level} RISK
                  </span>
                  <div className="flex items-center gap-1 text-[10px] text-white/30 font-mono">
                    <Clock size={10} />
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
                
                <h4 className="text-sm font-black text-white uppercase mb-1">{alert.sensor_id}</h4>
                <div className="flex items-center gap-4 text-[10px] text-white/50 uppercase font-bold">
                  <span className="flex items-center gap-1">
                    <MapPin size={10} />
                    Codrii Region
                  </span>
                  <span>Trigger: {alert.triggered_by}</span>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default AlertTimeline;

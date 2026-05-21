import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Rectangle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { InferenceResult } from '../hooks/useWebSocket';

// Fix for default marker icons in Leaflet + React
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface SensorMapProps {
  sensorStates: Record<string, InferenceResult>;
  onSelectSensor: (sensor_id: string) => void;
}

const getMarkerColor = (risk: string) => {
  switch (risk) {
    case 'HIGH': return '#ef4444';
    case 'MEDIUM': return '#f59e0b';
    default: return '#22c55e';
  }
};

const createCustomIcon = (color: string) => {
  return L.divIcon({
    className: 'custom-div-icon',
    html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.5);"></div>`,
    iconSize: [12, 12],
    iconAnchor: [6, 6]
  });
};

const SensorMap: React.FC<SensorMapProps> = ({ sensorStates, onSelectSensor }) => {
  const moldovaCenter: [number, number] = [47.0, 28.9];

  // Simple bounding boxes for Moldova regions for heatmap
  const REGION_BOUNDS: Record<string, [[number, number], [number, number]]> = {
    'Chisinau': [[46.9, 28.7], [47.1, 29.0]],
    'Codrii': [[47.0, 28.1], [47.3, 28.5]],
    'North': [[47.5, 27.5], [48.0, 28.3]],
    'South': [[45.6, 28.0], [46.2, 28.6]],
  };

  return (
    <div className="w-full h-full rounded-[2rem] overflow-hidden border border-white/10 shadow-2xl relative">
      <MapContainer 
        center={moldovaCenter} 
        zoom={8} 
        className="w-full h-full"
        style={{ background: '#111' }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        
        {/* Risk Heatmap Overlays */}
        {Object.entries(REGION_BOUNDS).map(([region, bounds]) => {
          // Find worst risk in this region
          const sensorsInRegion = Object.values(sensorStates).filter(s => s.region === region);
          let risk = 'LOW';
          if (sensorsInRegion.some(s => s.risk_level === 'HIGH')) risk = 'HIGH';
          else if (sensorsInRegion.some(s => s.risk_level === 'MEDIUM')) risk = 'MEDIUM';
          
          return (
            <Rectangle
              key={region}
              bounds={bounds}
              pathOptions={{
                fillColor: getMarkerColor(risk),
                fillOpacity: 0.15,
                color: 'transparent',
                weight: 0
              }}
            />
          );
        })}

        {/* Sensor Markers */}
        {Object.values(sensorStates).map((sensor) => (
          <Marker 
            key={sensor.sensor_id} 
            position={[sensor.lat, sensor.lon]}
            icon={createCustomIcon(getMarkerColor(sensor.risk_level))}
            eventHandlers={{
                click: () => onSelectSensor(sensor.sensor_id)
            }}
          >
            <Popup className="custom-popup">
              <div className="p-2">
                <h3 className="font-black text-black uppercase text-xs mb-1">{sensor.sensor_id}</h3>
                <p className="text-[10px] text-gray-500 uppercase font-bold">{sensor.region} Region</p>
                <div className="mt-2 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: getMarkerColor(sensor.risk_level) }} />
                  <span className="text-[10px] font-black uppercase" style={{ color: getMarkerColor(sensor.risk_level) }}>
                    {sensor.risk_level} RISK
                  </span>
                </div>
                <p className="text-[9px] text-gray-400 mt-1">Last update: {new Date(sensor.timestamp).toLocaleTimeString()}</p>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default SensorMap;

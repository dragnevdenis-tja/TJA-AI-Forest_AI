import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live';

export interface InferenceResult {
  event: 'inference_result';
  sensor_id: string;
  region: string;
  timestamp: string;
  audio_predictions: Record<string, number>;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  probabilities: Record<string, number>;
  alert_triggered: boolean;
  lat: number;
  lon: number;
}

export interface AlertEvent {
  event: 'alert';
  sensor_id: string;
  risk_level: string;
  timestamp: string;
}

export interface InitialStatus {
  event: 'initial_status';
  sensors: any[];
}

export type WSEvent = InferenceResult | AlertEvent | InitialStatus;

export const useWebSocket = () => {
  const [lastEvent, setLastEvent] = useState<WSEvent | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [sensorStates, setSensorStates] = useState<Record<string, InferenceResult>>({});
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);
  const reconnectDelayRef = useRef(1000);

  const connect = useCallback(() => {
    console.log('Connecting to WebSocket...');
    const ws = new WebSocket(WS_URL);
    socketRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
      reconnectDelayRef.current = 1000;
    };

    ws.onmessage = (event) => {
      const data: WSEvent = JSON.parse(event.data);
      setLastEvent(data);

      if (data.event === 'inference_result') {
        setSensorStates((prev) => ({
          ...prev,
          [data.sensor_id]: data
        }));
      } else if (data.event === 'initial_status') {
        // Initialize sensor states with registry data if needed
        const initialStates: Record<string, any> = {};
        data.sensors.forEach(s => {
            initialStates[s.id] = {
                sensor_id: s.id,
                region: s.region,
                lat: s.lat,
                lon: s.lng, // Backwards compatibility with registry
                risk_level: 'LOW',
                timestamp: new Date().toISOString()
            };
        });
        setSensorStates(initialStates);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
      
      // Auto-reconnect with exponential backoff
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 2, 30000);
        connect();
      }, reconnectDelayRef.current);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  return { lastEvent, isConnected, sensorStates };
};

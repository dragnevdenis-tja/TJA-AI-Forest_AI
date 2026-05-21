import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const fetchRiskHistory = async (sensor_id: string, limit: number = 50) => {
  try {
    const response = await api.get('/api/risk/history', {
      params: { sensor_id, limit }
    });
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to fetch risk history';
    throw new Error(message);
  }
};

export const analyzeAudio = async (audioBlob: Blob) => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.wav');
  // In a real scenario, we might add metadata fields here
  
  try {
    const response = await api.post('/api/audio/analyze', formData);
    return response.data;
  } catch (error: any) {
    const message = error.response?.data?.detail || error.message || 'Failed to analyze audio';
    throw new Error(message);
  }
};

export default api;

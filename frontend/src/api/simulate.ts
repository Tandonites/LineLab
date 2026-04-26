import type { DrawnStation, Prediction, TrainService } from '../App'

interface SimulateRequest {
  train_service: TrainService
  stations: {
    id: string
    name: string
    lat: number
    lon: number
    is_new: boolean
  }[]
}

export async function simulateNewLine(
  stations: DrawnStation[],
  trainService: TrainService
): Promise<Prediction> {
  const payload: SimulateRequest = {
    train_service: trainService,
    stations: stations.map(st => ({
      id: st.id,
      name: st.name,
      lat: st.lat,
      lon: st.lon,
      is_new: st.isNew,
    })),
  }

  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }

  return (await res.json()) as Prediction
}

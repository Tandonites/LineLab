import { useState, useCallback } from 'react'
import SubwayMap from './components/SubwayMap'
import LeftPanel from './components/LeftPanel'
import Toast from './components/Toast'

export interface DrawnStation {
  id: string
  name: string
  lat: number
  lon: number
  isNew: boolean
  lines: string[]
}

export interface NewStationDraft {
  lat: number
  lon: number
  x: number
  y: number
}

export interface AffectedStation {
  station_id: string
  name: string
  ridership_delta: number
  ridership_delta_pct: number
}

export interface Prediction {
  new_line_ridership: number
  peak_hour_ridership: number
  operational_cost_daily: number
  affected_lines: { line: string; delta_pct: number }[]
  affected_stations: AffectedStation[]
}

export type Mode = 'draw' | 'results'

export interface AppState {
  mode: Mode
  drawnLine: DrawnStation[]
  newStationDraft: NewStationDraft | null
  loading: boolean
  prediction: Prediction | null
  error: string | null
}

export default function App() {
  const [state, setState] = useState<AppState>({
    mode: 'draw',
    drawnLine: [],
    newStationDraft: null,
    loading: false,
    prediction: null,
    error: null,
  })

  const setMode = useCallback((mode: Mode) => {
    setState(s => ({ ...s, mode }))
  }, [])

  const addStation = useCallback((station: DrawnStation) => {
    setState(s => {
      if (s.drawnLine.find(st => st.id === station.id)) return s
      return { ...s, drawnLine: [...s.drawnLine, station] }
    })
  }, [])

  const removeStation = useCallback((id: string) => {
    setState(s => ({ ...s, drawnLine: s.drawnLine.filter(st => st.id !== id) }))
  }, [])

  const undoLast = useCallback(() => {
    setState(s => ({ ...s, drawnLine: s.drawnLine.slice(0, -1) }))
  }, [])

  const clearAll = useCallback(() => {
    setState(s => ({ ...s, drawnLine: [], prediction: null, mode: 'draw' }))
  }, [])

  const reorderLine = useCallback((newOrder: DrawnStation[]) => {
    setState(s => ({ ...s, drawnLine: newOrder }))
  }, [])

  const setNewStationDraft = useCallback((draft: NewStationDraft | null) => {
    setState(s => ({ ...s, newStationDraft: draft }))
  }, [])

  const commitNewStation = useCallback((name: string) => {
    setState(s => {
      if (!s.newStationDraft) return s
      const idx = s.drawnLine.filter(st => st.isNew).length
      const newStation: DrawnStation = {
        id: `new_${idx}`,
        name,
        lat: s.newStationDraft.lat,
        lon: s.newStationDraft.lon,
        isNew: true,
        lines: [],
      }
      return {
        ...s,
        drawnLine: [...s.drawnLine, newStation],
        newStationDraft: null,
      }
    })
  }, [])

  const cancelDraft = useCallback(() => {
    setState(s => ({ ...s, newStationDraft: null }))
  }, [])

  const predict = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stations: state.drawnLine.map(st => ({
            id: st.id,
            name: st.name,
            lat: st.lat,
            lon: st.lon,
            is_new: st.isNew,
          })),
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const prediction: Prediction = await res.json()
      setState(s => ({ ...s, loading: false, prediction, mode: 'results' }))
    } catch {
      setState(s => ({
        ...s,
        loading: false,
        error: 'Prediction failed — check that the backend is running.',
      }))
    }
  }, [state.drawnLine])

  const dismissError = useCallback(() => {
    setState(s => ({ ...s, error: null }))
  }, [])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0f1117]">
      <LeftPanel
        state={state}
        setMode={setMode}
        removeStation={removeStation}
        undoLast={undoLast}
        clearAll={clearAll}
        reorderLine={reorderLine}
        predict={predict}
      />
      <div className="flex-1 relative">
        <SubwayMap
          drawnLine={state.drawnLine}
          prediction={state.prediction}
          newStationDraft={state.newStationDraft}
          onStationClick={addStation}
          onMapRightClick={setNewStationDraft}
          onCommitNewStation={commitNewStation}
          onCancelDraft={cancelDraft}
        />
      </div>
      {state.error && (
        <Toast message={state.error} onDismiss={dismissError} />
      )}
    </div>
  )
}

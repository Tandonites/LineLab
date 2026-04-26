import { useState, useCallback } from 'react'
import SubwayMap from './components/SubwayMap'
import LeftPanel from './components/LeftPanel'
import Toast from './components/Toast'
import { simulateNewLine } from './api/simulate'

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
  route_comparison: {
    available: boolean
    existing_route_label: string
    origin_name: string
    destination_name: string
    first_train: string
    transfer_station: string | null
    second_train: string | null
    existing_travel_minutes: number
    new_route_minutes: number
    time_saved_minutes: number
  } | null
}

export type Mode = 'draw' | 'results'
export type TrainService = 'local' | 'express'

export interface AppState {
  mode: Mode
  drawnLine: DrawnStation[]
  newStationDraft: NewStationDraft | null
  loading: boolean
  prediction: Prediction | null
  error: string | null
  validationError: string | null
  trainService: TrainService
  showAllStations: boolean
}

const SERVICE_RULES: Record<TrainService, { minMiles: number; maxMiles: number }> = {
  local: { minMiles: 0.2, maxMiles: 0.75 },
  express: { minMiles: 0.5, maxMiles: 3 },
}

function haversineMiles(
  a: { lat: number; lon: number },
  b: { lat: number; lon: number }
): number {
  const R = 3958.8
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLon = ((b.lon - a.lon) * Math.PI) / 180
  const lat1 = (a.lat * Math.PI) / 180
  const lat2 = (b.lat * Math.PI) / 180
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2

  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x))
}

function validateLineSpacing(
  line: DrawnStation[],
  trainService: TrainService
): string | null {
  const { minMiles, maxMiles } = SERVICE_RULES[trainService]

  for (let i = 1; i < line.length; i += 1) {
    const miles = haversineMiles(line[i - 1], line[i])
    if (miles < minMiles) {
      return `${line[i - 1].name} to ${line[i].name} is ${miles.toFixed(2)} miles. ${trainService === 'local' ? 'Local' : 'Express'} stops must be at least ${minMiles.toFixed(1)} miles apart.`
    }
    if (miles > maxMiles) {
      return `${line[i - 1].name} to ${line[i].name} is ${miles.toFixed(2)} miles. ${trainService === 'local' ? 'Local' : 'Express'} stops must be no more than ${maxMiles.toFixed(1)} miles apart.`
    }
  }

  return null
}

export default function App() {
  const [state, setState] = useState<AppState>({
    mode: 'draw',
    drawnLine: [],
    newStationDraft: null,
    loading: false,
    prediction: null,
    error: null,
    validationError: null,
    trainService: 'local',
    showAllStations: true,
  })

  const setMode = useCallback((mode: Mode) => {
    setState(s => ({ ...s, mode }))
  }, [])

  const setTrainService = useCallback((trainService: TrainService) => {
    setState(s => {
      const validationError = validateLineSpacing(s.drawnLine, trainService)
      return {
        ...s,
        trainService,
        validationError,
        error: validationError,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
      }
    })
  }, [])

  const setShowAllStations = useCallback((showAllStations: boolean) => {
    setState(s => ({ ...s, showAllStations }))
  }, [])

  const addStation = useCallback((station: DrawnStation) => {
    setState(s => {
      if (s.drawnLine.find(st => st.id === station.id)) return s
      const nextLine = [...s.drawnLine, station]
      const validationError = validateLineSpacing(nextLine, s.trainService)
      if (validationError) {
        return { ...s, error: validationError, validationError }
      }
      return { ...s, drawnLine: nextLine, validationError: null, error: null }
    })
  }, [])

  const removeStation = useCallback((id: string) => {
    setState(s => {
      const drawnLine = s.drawnLine.filter(st => st.id !== id)
      const validationError = validateLineSpacing(drawnLine, s.trainService)
      return {
        ...s,
        drawnLine,
        validationError,
        error: validationError,
      }
    })
  }, [])

  const undoLast = useCallback(() => {
    setState(s => {
      const drawnLine = s.drawnLine.slice(0, -1)
      const validationError = validateLineSpacing(drawnLine, s.trainService)
      return {
        ...s,
        drawnLine,
        validationError,
        error: validationError,
      }
    })
  }, [])

  const clearAll = useCallback(() => {
    setState(s => ({
      ...s,
      drawnLine: [],
      prediction: null,
      mode: 'draw',
      validationError: null,
      error: null,
      showAllStations: true,
    }))
  }, [])

  const reorderLine = useCallback((newOrder: DrawnStation[]) => {
    setState(s => {
      const validationError = validateLineSpacing(newOrder, s.trainService)
      if (validationError) {
        return { ...s, error: validationError, validationError }
      }
      return { ...s, drawnLine: newOrder, validationError: null, error: null }
    })
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
      const nextLine = [...s.drawnLine, newStation]
      const validationError = validateLineSpacing(nextLine, s.trainService)
      if (validationError) {
        return {
          ...s,
          newStationDraft: null,
          error: validationError,
          validationError,
        }
      }
      return {
        ...s,
        drawnLine: nextLine,
        newStationDraft: null,
        validationError: null,
        error: null,
      }
    })
  }, [])

  const cancelDraft = useCallback(() => {
    setState(s => ({ ...s, newStationDraft: null }))
  }, [])

  const predict = useCallback(async () => {
    const validationError = validateLineSpacing(state.drawnLine, state.trainService)
    if (validationError) {
      setState(s => ({ ...s, error: validationError, validationError }))
      return
    }

    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const prediction = await simulateNewLine(state.drawnLine, state.trainService)
      setState(s => ({
        ...s,
        loading: false,
        prediction,
        mode: 'results',
        validationError: null,
        showAllStations: false,
      }))
    } catch {
      setState(s => ({
        ...s,
        loading: false,
        prediction: null,
        mode: 'draw',
        error: 'Prediction failed — backend unavailable or returned invalid data.',
        showAllStations: true,
      }))
    }
  }, [state.drawnLine, state.trainService])

  const dismissError = useCallback(() => {
    setState(s => ({ ...s, error: null }))
  }, [])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0f1117]">
      <LeftPanel
        state={state}
        setMode={setMode}
        setTrainService={setTrainService}
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
          trainService={state.trainService}
          showAllStations={state.showAllStations}
          onToggleAllStations={setShowAllStations}
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

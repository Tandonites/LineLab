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
  operational_cost_monthly: number
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
  suggestionSummary: string | null
  originalLine: { drawnLine: DrawnStation[]; trainService: TrainService; prediction: Prediction | null } | null
  suggestedLine: { drawnLine: DrawnStation[]; trainService: TrainService; prediction: Prediction; summary: string } | null
  showingSuggestedLine: boolean
  error: string | null
  validationError: string | null
  trainService: TrainService
  showAllStations: boolean
}

interface SuggestionCandidate {
  stations: DrawnStation[]
  trainService: TrainService
}

interface ImportedStop {
  id?: string
  station_id?: string
  station_complex_id?: string
  name?: string
  lat?: number
  lon?: number
  lng?: number
  latitude?: number
  longitude?: number
  isNew?: boolean
  is_new?: boolean
  lines?: string[]
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

function normalizeTrainService(value: unknown): TrainService {
  return value === 'express' ? 'express' : 'local'
}

function normalizeImportedLine(payload: unknown): {
  trainService: TrainService
  drawnLine: DrawnStation[]
} {
  const root = payload as
    | {
        train_service?: unknown
        trainService?: unknown
        stations?: unknown
        stops?: unknown
        line?: unknown
      }
    | ImportedStop[]

  const rawStops = Array.isArray(root)
    ? root
    : Array.isArray(root.stations)
      ? root.stations
      : Array.isArray(root.stops)
        ? root.stops
        : Array.isArray(root.line)
          ? root.line
          : null

  if (!rawStops || rawStops.length < 2) {
    throw new Error('JSON must include at least two stops in `stations`, `stops`, or the root array.')
  }

  const trainService = normalizeTrainService(
    Array.isArray(root) ? undefined : root.train_service ?? root.trainService
  )

  const drawnLine = rawStops.map((rawStop, index) => {
    const stop = rawStop as ImportedStop
    const lat = stop.lat ?? stop.latitude
    const lon = stop.lon ?? stop.lng ?? stop.longitude

    if (typeof lat !== 'number' || typeof lon !== 'number') {
      throw new Error(`Stop ${index + 1} is missing numeric lat/lon coordinates.`)
    }

    return {
      id:
        stop.id ??
        stop.station_complex_id ??
        stop.station_id ??
        `import_${index}_${lat.toFixed(5)}_${lon.toFixed(5)}`,
      name: stop.name ?? `Imported Stop ${index + 1}`,
      lat,
      lon,
      isNew: stop.isNew ?? stop.is_new ?? !(stop.station_complex_id || stop.station_id),
      lines: Array.isArray(stop.lines) ? stop.lines.map(String) : [],
    }
  })

  return { trainService, drawnLine }
}

export default function App() {
  const [mapVersion, setMapVersion] = useState(0)
  const [state, setState] = useState<AppState>({
    mode: 'draw',
    drawnLine: [],
    newStationDraft: null,
    loading: false,
    prediction: null,
    suggestionSummary: null,
    originalLine: null,
    suggestedLine: null,
    showingSuggestedLine: false,
    error: null,
    validationError: null,
    trainService: 'local',
    showAllStations: true,
  })

  const setMode = useCallback((mode: Mode) => {
    setState(s => ({
      ...s,
      mode,
      showAllStations: mode === 'draw' ? true : s.showAllStations,
    }))
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
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
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
      return {
        ...s,
        drawnLine: nextLine,
        validationError: null,
        error: null,
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
      }
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
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
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
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
      }
    })
  }, [])

  const clearAll = useCallback(() => {
    setMapVersion(version => version + 1)
    setState(s => ({
      ...s,
      drawnLine: [],
      newStationDraft: null,
      prediction: null,
      suggestionSummary: null,
      originalLine: null,
      suggestedLine: null,
      showingSuggestedLine: false,
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
      return {
        ...s,
        drawnLine: newOrder,
        validationError: null,
        error: null,
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
      }
    })
  }, [])

  const setNewStationDraft = useCallback((draft: NewStationDraft | null) => {
    setState(s => ({
      ...s,
      newStationDraft: draft,
      mode: draft ? 'draw' : s.mode,
      showAllStations: draft ? true : s.showAllStations,
    }))
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
          suggestionSummary: null,
          originalLine: null,
          suggestedLine: null,
          showingSuggestedLine: false,
          prediction: null,
          mode: 'draw',
          showAllStations: true,
        }
      }
      return {
        ...s,
        drawnLine: nextLine,
        newStationDraft: null,
        validationError: null,
        error: null,
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        prediction: null,
        mode: 'draw',
        showAllStations: true,
      }
    })
  }, [])

  const cancelDraft = useCallback(() => {
    setState(s => ({ ...s, newStationDraft: null }))
  }, [])

  const runPrediction = useCallback(async (drawnLine: DrawnStation[], trainService: TrainService) => {
    const validationError = validateLineSpacing(drawnLine, trainService)
    if (validationError) {
      setState(s => ({ ...s, error: validationError, validationError }))
      return
    }

    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const prediction = await simulateNewLine(drawnLine, trainService)
      setState(s => ({
        ...s,
        drawnLine,
        trainService,
        newStationDraft: null,
        loading: false,
        prediction,
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        mode: 'results',
        validationError: null,
        showAllStations: false,
      }))
    } catch {
      setState(s => ({
        ...s,
        loading: false,
        prediction: null,
        suggestionSummary: null,
        originalLine: null,
        suggestedLine: null,
        showingSuggestedLine: false,
        mode: 'draw',
        error: 'Prediction failed — backend unavailable or returned invalid data.',
        showAllStations: true,
      }))
    }
  }, [])

  const predict = useCallback(async () => {
    await runPrediction(state.drawnLine, state.trainService)
  }, [runPrediction, state.drawnLine, state.trainService])

  const importJsonLine = useCallback(async (file: File) => {
    try {
      const rawText = await file.text()
      const parsed = JSON.parse(rawText)
      const { trainService, drawnLine } = normalizeImportedLine(parsed)
      await runPrediction(drawnLine, trainService)
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Unable to import that JSON file.'
      setState(s => ({
        ...s,
        loading: false,
        error: `Import failed — ${message}`,
      }))
    }
  }, [runPrediction])

  const exportJsonLine = useCallback(() => {
    if (state.drawnLine.length < 2) return

    const payload = {
      train_service: state.trainService,
      stations: state.drawnLine.map(station => ({
        id: station.id,
        name: station.name,
        lat: station.lat,
        lon: station.lon,
        is_new: station.isNew,
        lines: station.lines,
      })),
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `linelab-${state.trainService}-${state.drawnLine.length}-stops.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [state.drawnLine, state.trainService])

  const suggestCheaperLine = useCallback(async () => {
    const validationError = validateLineSpacing(state.drawnLine, state.trainService)
    if (validationError) {
      setState(s => ({ ...s, error: validationError, validationError }))
      return
    }

    if (state.drawnLine.length < 2) {
      setState(s => ({ ...s, error: 'Add at least two stops before requesting a suggestion.' }))
      return
    }

    const dedupe = (candidates: SuggestionCandidate[]): SuggestionCandidate[] => {
      const seen = new Set<string>()
      const out: SuggestionCandidate[] = []
      for (const candidate of candidates) {
        const key = `${candidate.trainService}:${candidate.stations.map(st => st.id).join('>')}`
        if (seen.has(key)) continue
        seen.add(key)
        out.push(candidate)
      }
      return out
    }

    const buildCandidates = (): SuggestionCandidate[] => {
      const base = state.drawnLine
      const candidates: SuggestionCandidate[] = []
      const alternateService: TrainService = state.trainService === 'local' ? 'express' : 'local'

      if (!validateLineSpacing(base, alternateService)) {
        candidates.push({ stations: base, trainService: alternateService })
      }

      if (base.length >= 3) {
        for (let i = 1; i < base.length - 1; i += 1) {
          const pruned = base.filter((_, idx) => idx !== i)
          if (!validateLineSpacing(pruned, state.trainService)) {
            candidates.push({ stations: pruned, trainService: state.trainService })
          }
          if (!validateLineSpacing(pruned, alternateService)) {
            candidates.push({ stations: pruned, trainService: alternateService })
          }
        }
      }

      if (base.length >= 5) {
        const mids = base.slice(1, -1).filter((_, idx) => idx % 2 === 0)
        const everyOther = [base[0], ...mids, base[base.length - 1]]
        if (!validateLineSpacing(everyOther, state.trainService)) {
          candidates.push({ stations: everyOther, trainService: state.trainService })
        }
        if (!validateLineSpacing(everyOther, alternateService)) {
          candidates.push({ stations: everyOther, trainService: alternateService })
        }
      }

      return dedupe(candidates)
    }

    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const baseline = await simulateNewLine(state.drawnLine, state.trainService)
      const candidates = buildCandidates()

      if (candidates.length === 0) {
        setState(s => ({
          ...s,
          loading: false,
          error: 'No valid similar alternatives found for the current spacing rules.',
        }))
        return
      }

      const valid: Array<SuggestionCandidate & { prediction: Prediction }> = []
      for (const candidate of candidates) {
        try {
          const prediction = await simulateNewLine(candidate.stations, candidate.trainService)
          valid.push({ ...candidate, prediction })
        } catch (error) {
          console.warn('Skipping cheaper-line candidate after failed simulation', {
            candidate,
            error,
          })
        }
      }

      if (valid.length === 0) {
        setState(s => ({
          ...s,
          loading: false,
          suggestionSummary: null,
          error: 'Unable to evaluate alternate lines right now. Try again in a moment.',
        }))
        return
      }

      const similarCheaper = valid
        .filter(item => {
          const ridershipRatio = item.prediction.new_line_ridership / Math.max(1, baseline.new_line_ridership)
          const similar = ridershipRatio >= 0.8 && ridershipRatio <= 1.2
          const cheaper = item.prediction.operational_cost_monthly < baseline.operational_cost_monthly * 0.98
          return similar && cheaper
        })
        .sort((a, b) => {
          const costDelta =
            a.prediction.operational_cost_monthly - b.prediction.operational_cost_monthly
          if (costDelta !== 0) return costDelta

          const aRidershipGap = Math.abs(
            a.prediction.new_line_ridership - baseline.new_line_ridership
          )
          const bRidershipGap = Math.abs(
            b.prediction.new_line_ridership - baseline.new_line_ridership
          )
          return aRidershipGap - bRidershipGap
        })

      const best = similarCheaper[0] ?? null

      if (!best) {
        setState(s => ({
          ...s,
          loading: false,
          suggestionSummary: null,
          error: 'No cheaper similar line found. Try adding/removing a stop and retry.',
        }))
        return
      }

      const stopDelta = best.stations.length - state.drawnLine.length
      const baselineCost = baseline.operational_cost_monthly
      const costDeltaPct = ((best.prediction.operational_cost_monthly - baselineCost) / Math.max(1, baselineCost)) * 100
      const ridershipDeltaPct =
        ((best.prediction.new_line_ridership - baseline.new_line_ridership) / Math.max(1, baseline.new_line_ridership)) * 100

      const parts: string[] = []
      if (best.trainService !== state.trainService) {
        parts.push(`switched to ${best.trainService}`)
      }
      if (stopDelta !== 0) {
        parts.push(stopDelta < 0 ? `removed ${Math.abs(stopDelta)} stop${Math.abs(stopDelta) === 1 ? '' : 's'}` : `added ${stopDelta} stop${stopDelta === 1 ? '' : 's'}`)
      }
      if (parts.length === 0) {
        parts.push('kept same stop count and service pattern')
      }

      const summary = `${parts.join(', ')} · cost ${costDeltaPct.toFixed(1)}% · ridership ${ridershipDeltaPct >= 0 ? '+' : ''}${ridershipDeltaPct.toFixed(1)}%`

      setState(s => ({
        ...s,
        loading: false,
        drawnLine: best.stations,
        trainService: best.trainService,
        prediction: best.prediction,
        suggestionSummary: summary,
        originalLine: {
          drawnLine: state.drawnLine,
          trainService: state.trainService,
          prediction: baseline,
        },
        suggestedLine: {
          drawnLine: best.stations,
          trainService: best.trainService,
          prediction: best.prediction,
          summary,
        },
        showingSuggestedLine: true,
        mode: 'results',
        showAllStations: false,
        validationError: null,
        error: null,
      }))
    } catch {
      setState(s => ({
        ...s,
        loading: false,
        suggestionSummary: null,
        error: 'Line suggestion failed — backend unavailable or returned invalid data.',
      }))
    }
  }, [state.drawnLine, state.trainService])

  const toggleSuggestedLineView = useCallback(() => {
    setState(s => {
      if (!s.originalLine || !s.suggestedLine) return s

      if (s.showingSuggestedLine) {
        return {
          ...s,
          drawnLine: s.originalLine.drawnLine,
          trainService: s.originalLine.trainService,
          prediction: s.originalLine.prediction,
          suggestionSummary: null,
          showingSuggestedLine: false,
          mode: 'results',
          showAllStations: false,
          error: null,
        }
      }

      return {
        ...s,
        drawnLine: s.suggestedLine.drawnLine,
        trainService: s.suggestedLine.trainService,
        prediction: s.suggestedLine.prediction,
        suggestionSummary: s.suggestedLine.summary,
        showingSuggestedLine: true,
        mode: 'results',
        showAllStations: false,
        error: null,
      }
    })
  }, [])

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
        importJsonLine={importJsonLine}
        exportJsonLine={exportJsonLine}
        suggestCheaperLine={suggestCheaperLine}
        toggleSuggestedLineView={toggleSuggestedLineView}
      />
      <div className="flex-1 relative">
        <SubwayMap
          key={mapVersion}
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

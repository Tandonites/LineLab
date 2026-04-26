import { useEffect, useRef, useCallback, useState } from 'react'
import L from 'leaflet'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import type { DrawnStation, NewStationDraft, Prediction, TrainService } from '../App'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

const LINE_COLORS: Record<string, string> = {
  blue: '#0039A6',
  orange: '#FF6319',
  yellow: '#FCCC0A',
  red: '#EE352E',
  green: '#00933C',
  brown: '#996633',
  grey: '#A7A9AC',
  lime: '#6CBE45',
  purple: '#B933AD',
  shuttle: '#808183',
  other: '#A7A9AC',
}

const LINE_GROUP: Record<string, string> = {
  A: 'blue', C: 'blue', E: 'blue',
  B: 'orange', D: 'orange', F: 'orange', M: 'orange',
  N: 'yellow', Q: 'yellow', R: 'yellow', W: 'yellow',
  '1': 'red', '2': 'red', '3': 'red',
  '4': 'green', '5': 'green', '6': 'green',
  J: 'brown', Z: 'brown',
  L: 'grey', G: 'lime', '7': 'purple', S: 'shuttle',
}

function stationColor(lines: string[]): string {
  for (const line of lines) {
    const group = LINE_GROUP[line]
    if (group && LINE_COLORS[group]) return LINE_COLORS[group]
  }
  return LINE_COLORS.other
}

interface StationData {
  station_complex_id: string
  name: string
  lines: string[]
  borough: string
  lat: number
  lon: number
  total_ridership: number
}

interface Props {
  drawnLine: DrawnStation[]
  prediction: Prediction | null
  trainService: TrainService
  showAllStations: boolean
  onToggleAllStations: (show: boolean) => void
  newStationDraft: NewStationDraft | null
  onStationClick: (station: DrawnStation) => void
  onMapRightClick: (draft: NewStationDraft) => void
  onCommitNewStation: (name: string) => void
  onCancelDraft: () => void
}

const NYC_CENTER: [number, number] = [40.741, -73.94]
const NYC_ZOOM = 12
const NYC_MIN_ZOOM = 11
const NYC_MAX_BOUNDS = L.latLngBounds(
  [40.54, -74.15],
  [40.92, -73.68]
)
const NYC_VIEW_BOUNDS = L.latLngBounds(
  [40.54, -74.15],
  [40.92, -73.68]
)
const RESULT_HEAT_PANE = 'result-heat-pane'
const RESULT_BUBBLE_PANE = 'result-bubble-pane'
const ROUTE_PANE = 'route-pane'

const SERVICE_RADII_METERS: Record<TrainService, { min: number; max: number }> = {
  local: { min: 321.87, max: 1207.01 },
  express: { min: 804.67, max: 4828.03 },
}

function impactColor(delta: number): string {
  return delta >= 0 ? '#10b981' : '#ef4444'
}

function markerDisplayForZoom(zoom: number): { radius: number; fillOpacity: number; weight: number } {
  if (zoom <= 10) return { radius: 3.8, fillOpacity: 0.5, weight: 0.8 }
  if (zoom <= 11) return { radius: 4.8, fillOpacity: 0.66, weight: 0.9 }
  if (zoom <= 12) return { radius: 5.6, fillOpacity: 0.8, weight: 1 }
  return { radius: 6.4, fillOpacity: 0.92, weight: 1.1 }
}

export default function SubwayMap({
  drawnLine,
  prediction,
  trainService,
  showAllStations,
  onToggleAllStations,
  newStationDraft,
  onStationClick,
  onMapRightClick,
  onCommitNewStation,
  onCancelDraft,
}: Props) {
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const stationMarkersRef = useRef<Map<string, L.CircleMarker>>(new Map())
  const drawnPolylineRef = useRef<L.Polyline | null>(null)
  const stopRangeRef = useRef<L.Circle[]>([])
  const newStationMarkersRef = useRef<L.Marker[]>([])
  const routeStopMarkersRef = useRef<L.CircleMarker[]>([])
  const resultHeatRef = useRef<L.Circle[]>([])
  const resultBubblesRef = useRef<L.CircleMarker[]>([])
  const [stationsData, setStationsData] = useState<StationData[]>([])
  const [draftName, setDraftName] = useState('')

  useEffect(() => {
    fetch('/data/stations.json')
      .then(response => response.json())
      .then((data: StationData[]) => setStationsData(data))
      .catch(console.error)
  }, [])

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: NYC_CENTER,
      zoom: NYC_ZOOM,
      minZoom: NYC_MIN_ZOOM,
      zoomControl: false,
      maxBounds: NYC_MAX_BOUNDS,
      maxBoundsViscosity: 1,
      preferCanvas: true,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '© OpenStreetMap © CARTO',
      subdomains: 'abcd',
      maxZoom: 18,
    }).addTo(map)

    map.setView(NYC_CENTER, NYC_ZOOM)
    L.control.zoom({ position: 'bottomright' }).addTo(map)

    map.createPane(RESULT_HEAT_PANE)
    map.getPane(RESULT_HEAT_PANE)!.style.zIndex = '430'
    map.getPane(RESULT_HEAT_PANE)!.style.pointerEvents = 'none'
    map.createPane(RESULT_BUBBLE_PANE)
    map.getPane(RESULT_BUBBLE_PANE)!.style.zIndex = '460'
    map.createPane(ROUTE_PANE)
    map.getPane(ROUTE_PANE)!.style.zIndex = '490'
    map.getPane(ROUTE_PANE)!.style.pointerEvents = 'none'

    map.on('contextmenu', (e: L.LeafletMouseEvent) => {
      if (!NYC_VIEW_BOUNDS.contains(e.latlng)) return
      const containerPoint = map.latLngToContainerPoint(e.latlng)
      onMapRightClick({
        lat: e.latlng.lat,
        lon: e.latlng.lng,
        x: containerPoint.x,
        y: containerPoint.y,
      })
    })

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map || stationsData.length === 0) return
    const markers = stationMarkersRef.current

    stationsData.forEach(station => {
      if (!NYC_VIEW_BOUNDS.contains([station.lat, station.lon])) return

      const color = stationColor(station.lines)
      const isSelected = drawnLine.some(item => item.id === station.station_complex_id)

      const marker = L.circleMarker([station.lat, station.lon], {
        radius: 6,
        fillColor: color,
        color: isSelected ? '#ffffff' : color,
        weight: isSelected ? 3 : 1,
        fillOpacity: 0.9,
        opacity: 1,
        bubblingMouseEvents: false,
      })

      marker.bindTooltip(
        `<div class="font-semibold">${station.name}</div><div class="text-xs text-slate-300">${station.lines.join(' ')}</div>`,
        { direction: 'top', className: 'leaflet-tooltip-dark' }
      )

      marker.on('click', () => {
        onStationClick({
          id: station.station_complex_id,
          name: station.name,
          lat: station.lat,
          lon: station.lon,
          isNew: false,
          lines: station.lines,
        })
      })

      marker.addTo(map)
      markers.set(station.station_complex_id, marker)
    })

    return () => {
      markers.forEach(marker => marker.remove())
      markers.clear()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stationsData])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const applyZoomStyle = () => {
      const zoom = map.getZoom()
      const { radius, fillOpacity, weight } = markerDisplayForZoom(zoom)
      const selectedIds = new Set(drawnLine.map(station => station.id))
      const affectedIds = new Set(
        prediction?.affected_stations.map(station => station.station_id) ?? []
      )

      stationMarkersRef.current.forEach((marker, id) => {
        const selected = selectedIds.has(id)
        const visible =
          showAllStations || !prediction || selectedIds.has(id) || affectedIds.has(id)
        marker.setStyle({
          radius: visible ? (selected ? radius + 1.25 : radius) : 0,
          fillOpacity: visible ? (selected ? 1 : fillOpacity) : 0,
          opacity: visible ? 1 : 0,
          color: selected ? '#ffffff' : (marker.options.fillColor as string),
          weight: visible ? (selected ? Math.max(2.6, weight + 1.1) : weight) : 0,
        })
      })
    }

    applyZoomStyle()
    map.on('zoom zoomend moveend', applyZoomStyle)
    return () => {
      map.off('zoom zoomend moveend', applyZoomStyle)
    }
  }, [drawnLine, prediction, showAllStations])

  useEffect(() => {
    if (drawnPolylineRef.current) {
      drawnPolylineRef.current.remove()
      drawnPolylineRef.current = null
    }
    routeStopMarkersRef.current.forEach(marker => marker.remove())
    routeStopMarkersRef.current = []

    const map = mapRef.current
    if (!map) return

    if (drawnLine.length < 2) {
      drawnLine.forEach((station, index) => {
        const marker = L.circleMarker([station.lat, station.lon], {
          pane: ROUTE_PANE,
          radius: index === 0 ? 9 : 7,
          fillColor: station.isNew ? '#fb923c' : '#111827',
          fillOpacity: 1,
          color: '#ffffff',
          weight: 3,
          opacity: 1,
          interactive: false,
        })
          .bindTooltip(station.name, { direction: 'top', className: 'leaflet-tooltip-dark' })
          .addTo(map)
        routeStopMarkersRef.current.push(marker)
      })
      return
    }

    const latlngs = drawnLine.map(station => [station.lat, station.lon] as [number, number])
    drawnPolylineRef.current = L.polyline(latlngs, {
      pane: ROUTE_PANE,
      color: '#7dd3fc',
      weight: 5,
      opacity: 0.98,
      dashArray: '10 6',
      lineCap: 'round',
      lineJoin: 'round',
      interactive: false,
    }).addTo(map)

    drawnLine.forEach((station, index) => {
      const isEndpoint = index === 0 || index === drawnLine.length - 1
      const marker = L.circleMarker([station.lat, station.lon], {
        pane: ROUTE_PANE,
        radius: isEndpoint ? 10 : 7.5,
        fillColor: station.isNew ? '#fb923c' : '#111827',
        fillOpacity: 1,
        color: '#ffffff',
        weight: isEndpoint ? 3.5 : 3,
        opacity: 1,
        interactive: false,
      })
        .bindTooltip(
          `${index + 1}. ${station.name}`,
          { direction: 'top', className: 'leaflet-tooltip-dark' }
        )
        .addTo(map)

      routeStopMarkersRef.current.push(marker)
    })
  }, [drawnLine])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    newStationMarkersRef.current.forEach(marker => marker.remove())
    newStationMarkersRef.current = []

    drawnLine
      .filter(station => station.isNew)
      .forEach(station => {
        const icon = L.divIcon({
          className: '',
          html: `<div style="width:16px;height:16px;background:#fb923c;border:2px solid #fff;transform:rotate(45deg);border-radius:3px;box-shadow:0 0 0 4px rgba(251,146,60,0.14)"></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        })
        const marker = L.marker([station.lat, station.lon], { icon })
          .bindTooltip(station.name, { direction: 'top', className: 'leaflet-tooltip-dark' })
          .addTo(map)
        newStationMarkersRef.current.push(marker)
      })
  }, [drawnLine])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    stopRangeRef.current.forEach(layer => layer.remove())
    stopRangeRef.current = []

    if (drawnLine.length === 0) return

    const anchor = drawnLine[drawnLine.length - 1]
    const serviceRadii = SERVICE_RADII_METERS[trainService]

    const maxRing = L.circle([anchor.lat, anchor.lon], {
      radius: serviceRadii.max,
      color: '#7dd3fc',
      weight: 1.5,
      opacity: 0.75,
      dashArray: '7 6',
      fillColor: '#38bdf8',
      fillOpacity: 0.08,
      interactive: false,
    }).addTo(map)

    const minRing = L.circle([anchor.lat, anchor.lon], {
      radius: serviceRadii.min,
      color: trainService === 'express' ? '#fb923c' : '#fbbf24',
      weight: 1.25,
      opacity: 0.85,
      dashArray: '3 6',
      fillColor: trainService === 'express' ? '#ea580c' : '#f59e0b',
      fillOpacity: 0.08,
      interactive: false,
    }).addTo(map)

    stopRangeRef.current = [maxRing, minRing]
  }, [drawnLine, trainService])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    resultHeatRef.current.forEach(layer => layer.remove())
    resultHeatRef.current = []
    resultBubblesRef.current.forEach(layer => layer.remove())
    resultBubblesRef.current = []

    if (!prediction) return

    const maxAbs = Math.max(...prediction.affected_stations.map(station => Math.abs(station.ridership_delta)), 1)

    prediction.affected_stations.forEach(affected => {
      const marker = stationMarkersRef.current.get(affected.station_id)
      if (!marker) return

      const pct = Math.abs(affected.ridership_delta) / maxAbs
      const color = impactColor(affected.ridership_delta)
      const sign = affected.ridership_delta >= 0 ? '+' : ''
      const heatRadiusMeters = 520 + pct * 1200
      const bubbleRadius = 7 + pct * 15

      const outerGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.08 + pct * 0.08,
      }).addTo(map)

      const innerGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters * 0.55,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.12 + pct * 0.12,
      }).addTo(map)

      const coreGlow = L.circle(marker.getLatLng(), {
        radius: heatRadiusMeters * 0.24,
        pane: RESULT_HEAT_PANE,
        stroke: false,
        fillColor: color,
        fillOpacity: 0.18 + pct * 0.18,
      }).addTo(map)

      const bubble = L.circleMarker(marker.getLatLng(), {
        pane: RESULT_BUBBLE_PANE,
        radius: bubbleRadius,
        fillColor: color,
        fillOpacity: 0.46,
        color: '#f8fafc',
        weight: 2.4,
        opacity: 0.96,
      })
        .bindPopup(
          `<div class="text-sm font-semibold">${affected.name}</div><div class="text-xs">Δ ${sign}${affected.ridership_delta.toLocaleString()} riders/day</div><div class="text-xs">(${sign}${affected.ridership_delta_pct.toFixed(1)}%)</div>`
        )
        .addTo(map)

      resultHeatRef.current.push(outerGlow, innerGlow, coreGlow)
      resultBubblesRef.current.push(bubble)
    })
  }, [prediction])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onCancelDraft()
  }, [onCancelDraft])

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div ref={mapContainerRef} className="h-full w-full" />

      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-[linear-gradient(180deg,rgba(6,8,13,0.52),transparent)]" />

      <button
        className="absolute right-4 top-4 z-[600] rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(17,22,33,0.92),rgba(12,15,24,0.96))] px-4 py-2 text-sm font-medium text-white shadow-[0_14px_30px_rgba(0,0,0,0.24)] backdrop-blur-sm transition-all hover:border-white/18 hover:bg-[linear-gradient(180deg,rgba(24,31,45,0.96),rgba(15,20,31,0.98))]"
        onClick={() => mapRef.current?.setView(NYC_CENTER, NYC_ZOOM)}
      >
        Reset View
      </button>

      {prediction && (
        <button
          className={`absolute right-4 top-18 z-[600] rounded-full border px-4 py-2 text-sm font-medium shadow-[0_14px_30px_rgba(0,0,0,0.24)] backdrop-blur-sm transition-all ${
            showAllStations
              ? 'border-cyan-300/24 bg-cyan-400/12 text-cyan-100 hover:bg-cyan-400/18'
              : 'border-white/10 bg-[linear-gradient(180deg,rgba(17,22,33,0.92),rgba(12,15,24,0.96))] text-white hover:border-white/18'
          }`}
          onClick={() => onToggleAllStations(!showAllStations)}
        >
          {showAllStations ? 'Hide Other Stations' : 'Show All Stations'}
        </button>
      )}

      {prediction && (
        <div className="absolute bottom-6 left-6 z-[600] w-52 rounded-[1.35rem] border border-white/10 bg-[linear-gradient(180deg,rgba(14,18,28,0.92),rgba(9,12,20,0.96))] px-4 py-4 shadow-[0_20px_44px_rgba(0,0,0,0.3)] backdrop-blur-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Impact Map</p>
          <div className="mt-3 flex items-center gap-2 text-sm text-slate-200">
            <span className="inline-block h-3 w-3 rounded-full border border-white/60 bg-emerald-500/80" />
            Gains
          </div>
          <div className="mt-2 flex items-center gap-2 text-sm text-slate-200">
            <span className="inline-block h-3 w-3 rounded-full border border-white/60 bg-red-500/80" />
            Losses
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">
            Soft glow shows intensity while bright circles mark the affected stations.
          </p>
        </div>
      )}

      {drawnLine.length > 0 && (
        <div className="absolute bottom-6 right-24 z-[600] rounded-[1.35rem] border border-white/10 bg-[linear-gradient(180deg,rgba(14,18,28,0.92),rgba(9,12,20,0.96))] px-4 py-4 shadow-[0_20px_44px_rgba(0,0,0,0.3)] backdrop-blur-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">Stop Spacing</p>
          <div className="mt-3 flex items-center gap-2 text-sm text-slate-200">
            <span className="inline-block h-3 w-3 rounded-full border border-amber-300/70 bg-amber-500/70" />
            Minimum {trainService === 'local' ? '0.2 mi' : '0.5 mi'}
          </div>
          <div className="mt-2 flex items-center gap-2 text-sm text-slate-200">
            <span className="inline-block h-3 w-3 rounded-full border border-sky-300/70 bg-sky-500/70" />
            Maximum {trainService === 'local' ? '0.75 mi' : '3.0 mi'}
          </div>
        </div>
      )}

      {newStationDraft && (
        <div
          className="absolute z-[600] w-60 rounded-[1.25rem] border border-white/10 bg-[linear-gradient(180deg,rgba(18,23,35,0.96),rgba(11,15,24,0.98))] p-4 shadow-[0_22px_50px_rgba(0,0,0,0.34)] backdrop-blur-sm"
          style={{
            left: Math.min(newStationDraft.x + 12, window.innerWidth - 250),
            top: newStationDraft.y - 8,
          }}
          onKeyDown={handleKeyDown}
        >
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">New Station</p>
          <input
            autoFocus
            type="text"
            placeholder="Station name..."
            value={draftName}
            onChange={e => setDraftName(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') {
                setDraftName('')
                onCancelDraft()
              }
              if (e.key === 'Enter' && draftName.trim()) {
                onCommitNewStation(draftName.trim())
                setDraftName('')
              }
            }}
            className="mb-3 mt-3 w-full rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-white outline-none placeholder:text-slate-500 focus:border-cyan-400/50"
          />
          <div className="flex gap-2">
            <button
              disabled={!draftName.trim()}
              onClick={() => {
                onCommitNewStation(draftName.trim())
                setDraftName('')
              }}
              className="flex-1 rounded-xl bg-[linear-gradient(135deg,#22d3ee,#2563eb)] py-2 text-sm font-medium text-white transition-all hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Add to Line
            </button>
            <button
              onClick={() => {
                setDraftName('')
                onCancelDraft()
              }}
              className="flex-1 rounded-xl border border-white/10 bg-white/[0.04] py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-white/[0.08]"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

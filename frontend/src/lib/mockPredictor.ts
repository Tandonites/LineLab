import type { DrawnStation, Prediction, TrainService } from '../App'

interface StationData {
  station_complex_id: string
  name: string
  lines: string[]
  borough: string
  lat: number
  lon: number
  total_ridership: number
}

const EARTH_RADIUS_KM = 6371
const KM_TO_MILES = 0.621371
const SERVICE_SPEED_MPH: Record<TrainService, number> = {
  local: 18,
  express: 28,
}

const TRANSFER_HUBS = [
  'Times Sq-42 St',
  '34 St-Herald Sq',
  '14 St-Union Sq',
  'Atlantic Av-Barclays Ctr',
  'Fulton St',
  'Jackson Hts-Roosevelt Av',
  'Court Sq',
  'Canal St',
]

let stationsCache: StationData[] | null = null

function toRadians(value: number): number {
  return (value * Math.PI) / 180
}

function haversineKm(
  a: { lat: number; lon: number },
  b: { lat: number; lon: number }
): number {
  const dLat = toRadians(b.lat - a.lat)
  const dLon = toRadians(b.lon - a.lon)
  const lat1 = toRadians(a.lat)
  const lat2 = toRadians(b.lat)

  const term =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2

  return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(term), Math.sqrt(1 - term))
}

function distancePointToSegmentKm(
  point: { lat: number; lon: number },
  start: { lat: number; lon: number },
  end: { lat: number; lon: number }
): number {
  const avgLat = toRadians((start.lat + end.lat + point.lat) / 3)
  const lonScale = Math.cos(avgLat) * 111.32
  const latScale = 110.574

  const px = point.lon * lonScale
  const py = point.lat * latScale
  const sx = start.lon * lonScale
  const sy = start.lat * latScale
  const ex = end.lon * lonScale
  const ey = end.lat * latScale

  const dx = ex - sx
  const dy = ey - sy
  const segmentLengthSquared = dx * dx + dy * dy

  if (segmentLengthSquared === 0) {
    return Math.hypot(px - sx, py - sy)
  }

  const t = Math.max(
    0,
    Math.min(1, ((px - sx) * dx + (py - sy) * dy) / segmentLengthSquared)
  )

  const nearestX = sx + t * dx
  const nearestY = sy + t * dy
  return Math.hypot(px - nearestX, py - nearestY)
}

function totalRouteKm(line: DrawnStation[]): number {
  let total = 0
  for (let i = 1; i < line.length; i += 1) {
    total += haversineKm(line[i - 1], line[i])
  }
  return total
}

async function loadStations(): Promise<StationData[]> {
  if (stationsCache) return stationsCache

  const response = await fetch('/data/stations.json')
  if (!response.ok) {
    throw new Error(`Unable to load station data: HTTP ${response.status}`)
  }

  const data = (await response.json()) as StationData[]
  stationsCache = data
  return data
}

function pickTransferStation(
  stations: StationData[],
  start: DrawnStation,
  end: DrawnStation
): StationData | null {
  const startSet = new Set(start.lines)
  const endSet = new Set(end.lines)

  const hub = stations.find(station => {
    const hasStartLine = station.lines.some(line => startSet.has(line))
    const hasEndLine = station.lines.some(line => endSet.has(line))
    return hasStartLine && hasEndLine && TRANSFER_HUBS.includes(station.name)
  })

  if (hub) return hub

  return (
    stations.find(station => {
      const hasStartLine = station.lines.some(line => startSet.has(line))
      const hasEndLine = station.lines.some(line => endSet.has(line))
      return hasStartLine && hasEndLine
    }) ?? null
  )
}

export async function generateMockRouteComparison(
  drawnLine: DrawnStation[],
  trainService: TrainService
): Promise<Prediction['route_comparison']> {
  const stations = await loadStations()
  const start = drawnLine[0]
  const end = drawnLine[drawnLine.length - 1]

  if (!start || !end || start.isNew || end.isNew) {
    return null
  }

  const routeMiles = totalRouteKm(drawnLine) * KM_TO_MILES
  const newRouteMinutes = Math.max(
    7,
    Math.round((routeMiles / SERVICE_SPEED_MPH[trainService]) * 60 + drawnLine.length * 1.2)
  )

  const sharedLine = start.lines.find(line => end.lines.includes(line))
  const transferStation = sharedLine ? null : pickTransferStation(stations, start, end)
  const firstTrain = sharedLine ?? start.lines[0] ?? 'subway'
  const secondTrain = sharedLine
    ? null
    : transferStation?.lines.find(line => end.lines.includes(line)) ??
      end.lines[0] ??
      'subway'

  const existingRouteMinutes = sharedLine
    ? Math.max(newRouteMinutes + 4, Math.round(routeMiles * 3.6 + 9))
    : Math.max(newRouteMinutes + 9, Math.round(routeMiles * 4.4 + 14))

  return {
    available: true,
    existing_route_label: sharedLine
      ? `${sharedLine} train direct from ${start.name}`
      : `${firstTrain} to ${transferStation?.name ?? 'transfer hub'}, then ${secondTrain}`,
    origin_name: start.name,
    destination_name: end.name,
    first_train: firstTrain,
    transfer_station: transferStation?.name ?? null,
    second_train: secondTrain,
    existing_travel_minutes: existingRouteMinutes,
    new_route_minutes: newRouteMinutes,
    time_saved_minutes: Math.max(1, existingRouteMinutes - newRouteMinutes),
  }
}

export async function generateMockPrediction(
  drawnLine: DrawnStation[],
  trainService: TrainService
): Promise<Prediction> {
  const stations = await loadStations()
  const routeKm = totalRouteKm(drawnLine)

  const nearbyStations = stations
    .map(station => {
      let minDistanceKm = Infinity

      for (let i = 1; i < drawnLine.length; i += 1) {
        const segmentDistance = distancePointToSegmentKm(
          station,
          drawnLine[i - 1],
          drawnLine[i]
        )
        if (segmentDistance < minDistanceKm) {
          minDistanceKm = segmentDistance
        }
      }

      return { station, minDistanceKm }
    })
    .filter(({ minDistanceKm }) => minDistanceKm <= 1.8)
    .sort((a, b) => a.minDistanceKm - b.minDistanceKm)
    .slice(0, 18)

  const routeDemand = nearbyStations.reduce(
    (sum, { station }) => sum + station.total_ridership,
    0
  )

  const newLineRidership = Math.max(
    18000,
    Math.round(
      routeDemand * (trainService === 'express' ? 0.16 : 0.12) +
        routeKm * (trainService === 'express' ? 2100 : 1800) +
        drawnLine.length * 900
    )
  )

  const peakHourRidership = Math.round(
    newLineRidership * (trainService === 'express' ? 0.18 : 0.16)
  )
  const operationalCostDaily = Math.round(
    routeKm * (trainService === 'express' ? 240000 : 185000) +
      drawnLine.length * 22000
  )

  const lineImpact = new Map<string, number>()

  const affectedStations = nearbyStations.slice(0, 10).map(({ station, minDistanceKm }) => {
    const proximityFactor = Math.max(0.12, 1 - minDistanceKm / 1.8)
    const magnitude = Math.round(station.total_ridership * proximityFactor * 0.09)
    const direction = station.lines.length > 2 ? -1 : 1
    const ridershipDelta = direction * magnitude
    const ridershipDeltaPct = direction * Math.max(1.5, proximityFactor * 11)

    station.lines.forEach(line => {
      const current = lineImpact.get(line) ?? 0
      lineImpact.set(line, current + ridershipDeltaPct / station.lines.length)
    })

    return {
      station_id: station.station_complex_id,
      name: station.name,
      ridership_delta: ridershipDelta,
      ridership_delta_pct: ridershipDeltaPct,
    }
  })

  const affectedLines = Array.from(lineImpact.entries())
    .map(([line, delta_pct]) => ({
      line,
      delta_pct: Number(delta_pct.toFixed(1)),
    }))
    .sort((a, b) => Math.abs(b.delta_pct) - Math.abs(a.delta_pct))
    .slice(0, 6)

  return {
    new_line_ridership: newLineRidership,
    peak_hour_ridership: peakHourRidership,
    operational_cost_daily: operationalCostDaily,
    affected_lines: affectedLines,
    affected_stations: affectedStations,
    route_comparison: await generateMockRouteComparison(drawnLine, trainService),
  }
}

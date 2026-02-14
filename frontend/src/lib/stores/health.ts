/**
 * Enhanced health store with historical tracking for uptime charts
 */

import { writable, derived, get } from 'svelte/store';
import type { ServiceHealthResponse, ServiceCheck } from '$lib/types/api';
import { serviceHealth } from './dashboard';

interface HistoricalDataPoint {
	timestamp: string;
	services: Record<string, { status: string; duration_ms: number }>;
}

interface HealthHistory {
	dataPoints: HistoricalDataPoint[];
	maxPoints: number;
}

const HISTORY_MAX_POINTS = 100;

function createHealthHistoryStore() {
	const { subscribe, update } = writable<HealthHistory>({
		dataPoints: [],
		maxPoints: HISTORY_MAX_POINTS
	});

	// Subscribe to serviceHealth and add historical data points
	serviceHealth.subscribe((data) => {
		if (!data) return;

		const dataPoint: HistoricalDataPoint = {
			timestamp: new Date().toISOString(),
			services: {}
		};

		data.service_checks.forEach((check) => {
			dataPoint.services[check.service] = {
				status: check.status,
				duration_ms: check.duration_ms
			};
		});

		update((history) => {
			const newPoints = [...history.dataPoints, dataPoint];
			if (newPoints.length > history.maxPoints) {
				newPoints.shift();
			}
			return { ...history, dataPoints: newPoints };
		});
	});

	return { subscribe };
}

// Service name mapping
export const serviceNames: Record<string, string> = {
	alpha_vantage: 'Alpha Vantage',
	marketaux: 'Marketaux',
	alpaca: 'Alpaca',
	llm: 'LLM Provider',
	finnhub: 'Finnhub',
	circuit_breakers: 'Circuit Breakers'
};

export const healthHistory = createHealthHistoryStore();

// Derived store for uptime calculations
export const uptimeMetrics = derived(
	[serviceHealth, healthHistory],
	([$serviceHealth, $healthHistory]) => {
		if (!$serviceHealth) return null;

		const services: Record<string, {
			name: string;
			status: string;
			duration_ms: number;
			uptime_percent: number;
			checked_at: string;
			message: string;
		}> = {};

		$serviceHealth.service_checks.forEach((check) => {
			const serviceKey = check.service;
			const displayName = serviceNames[serviceKey] || serviceKey;

			// Calculate uptime from history
			let uptimePercent = 100;
			if ($healthHistory.dataPoints.length > 0) {
				const healthyCount = $healthHistory.dataPoints.filter(
					(dp) => dp.services[serviceKey]?.status === 'HEALTHY'
				).length;
				const totalCount = $healthHistory.dataPoints.filter(
					(dp) => dp.services[serviceKey] !== undefined
				).length;
				uptimePercent = totalCount > 0 ? (healthyCount / totalCount) * 100 : 100;
			}

			services[serviceKey] = {
				name: displayName,
				status: check.status,
				duration_ms: check.duration_ms,
				uptime_percent: uptimePercent,
				checked_at: check.checked_at,
				message: check.message
			};
		});

		return {
			services,
			overall_status: $serviceHealth.overall_status,
			total_services: Object.keys(services).length,
			healthy_services: Object.values(services).filter((s) => s.status === 'HEALTHY').length,
			avg_duration: Object.values(services).reduce((sum, s) => sum + s.duration_ms, 0) / Object.values(services).length,
			overall_uptime: Object.values(services).reduce((sum, s) => sum + s.uptime_percent, 0) / Object.values(services).length
		};
	}
);

// Derived store for degradation timeline events
export const degradationTimeline = derived(
	healthHistory,
	($healthHistory) => {
		const events: {
			timestamp: string;
			service: string;
			from_status: string;
			to_status: string;
			duration?: number;
		}[] = [];

		const points = $healthHistory.dataPoints;
		if (points.length < 2) return events;

		// Track status transitions
		for (let i = 1; i < points.length; i++) {
			const prev = points[i - 1];
			const curr = points[i];

			Object.keys(curr.services).forEach((serviceKey) => {
				const prevStatus = prev.services[serviceKey]?.status;
				const currStatus = curr.services[serviceKey]?.status;

				if (prevStatus && currStatus && prevStatus !== currStatus) {
					events.push({
						timestamp: curr.timestamp,
						service: serviceNames[serviceKey] || serviceKey,
						from_status: prevStatus,
						to_status: currStatus
					});
				}
			});
		}

		return events.reverse().slice(0, 50); // Last 50 events
	}
);

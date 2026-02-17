/**
 * Discovery insights store
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type { ActiveDiscoveryResponse, DiscoveryInsightsResponse } from '$lib/types/api';

export const discoveryInsights = writable<DiscoveryInsightsResponse | null>(null);
export const activeDiscovery = writable<ActiveDiscoveryResponse | null>(null);

export async function fetchDiscoveryInsights(): Promise<void> {
	try {
		const data = await api.getDiscoveryInsights();
		discoveryInsights.set(data);
	} catch (error) {
		console.error('Failed to fetch discovery insights:', error);
		discoveryInsights.set(null);
	}
}

export async function fetchActiveDiscovery(
	sourceFilter: 'all' | 'batch' | 'continuous' = 'all'
): Promise<void> {
	try {
		const data = await api.getActiveDiscovery(sourceFilter);
		activeDiscovery.set(data);
	} catch (error) {
		console.error('Failed to fetch active discovery:', error);
		activeDiscovery.set(null);
	}
}

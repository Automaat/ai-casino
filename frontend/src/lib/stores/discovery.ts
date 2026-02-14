/**
 * Discovery insights store
 */

import { writable } from 'svelte/store';
import { api } from '$lib/api/client';
import type { DiscoveryInsightsResponse } from '$lib/types/api';

export const discoveryInsights = writable<DiscoveryInsightsResponse | null>(null);

export async function fetchDiscoveryInsights(): Promise<void> {
	try {
		const data = await api.getDiscoveryInsights();
		discoveryInsights.set(data);
	} catch (error) {
		console.error('Failed to fetch discovery insights:', error);
		discoveryInsights.set(null);
	}
}

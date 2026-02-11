/**
 * Config value formatting utilities
 */

export interface FormattedValue {
	type: 'boolean' | 'string' | 'number' | 'list' | 'dict' | 'null';
	display: string;
	color?: string;
}

export function formatConfigValue(value: unknown): FormattedValue {
	if (value === null || value === undefined) {
		return {
			type: 'null',
			display: 'None',
			color: 'text-slate-500'
		};
	}

	if (typeof value === 'boolean') {
		return {
			type: 'boolean',
			display: value ? '✓' : '✗',
			color: value ? 'text-green-400' : 'text-red-400'
		};
	}

	if (Array.isArray(value)) {
		if (value.length === 0) {
			return {
				type: 'list',
				display: '[]',
				color: 'text-slate-500'
			};
		}
		return {
			type: 'list',
			display: JSON.stringify(value)
		};
	}

	if (typeof value === 'object') {
		const keys = Object.keys(value);
		return {
			type: 'dict',
			display: `${keys.length} items`,
			color: 'text-slate-500'
		};
	}

	return {
		type: typeof value as 'string' | 'number',
		display: String(value)
	};
}

export function getSectionEnabled(data: Record<string, unknown>): boolean {
	if ('enabled' in data) {
		return Boolean(data.enabled);
	}
	return false;
}

export function sortConfigKeys(data: Record<string, unknown>): [string, unknown][] {
	const entries = Object.entries(data);
	return entries.sort((a, b) => {
		const aIsEnabled = a[0] === 'enabled';
		const bIsEnabled = b[0] === 'enabled';

		if (aIsEnabled && !bIsEnabled) return -1;
		if (!aIsEnabled && bIsEnabled) return 1;

		return a[0].localeCompare(b[0]);
	});
}

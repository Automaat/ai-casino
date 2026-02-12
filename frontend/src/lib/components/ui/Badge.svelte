<script lang="ts">
	import type { Signal } from '$lib/types/api';
	import type { Snippet } from 'svelte';

	interface Props {
		variant: Signal | 'success' | 'error' | 'warning' | 'info' | 'neutral';
		children: Snippet;
	}

	let { variant, children }: Props = $props();

	const variantStyles: Record<string, string> = {
		BUY: 'bg-green-50 text-green-700 border-green-300',
		SELL: 'bg-red-50 text-red-700 border-red-300',
		HOLD: 'bg-gray-50 text-gray-700 border-gray-300',
		success: 'bg-green-50 text-green-700 border-green-300',
		error: 'bg-red-50 text-red-700 border-red-300',
		warning: 'bg-yellow-50 text-yellow-700 border-yellow-300',
		info: 'bg-blue-50 text-blue-700 border-blue-300',
		neutral: 'bg-gray-50 text-gray-700 border-gray-300'
	};

	const classes = $derived(variantStyles[variant] || variantStyles.neutral);
</script>

<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border {classes}">
	{@render children()}
</span>

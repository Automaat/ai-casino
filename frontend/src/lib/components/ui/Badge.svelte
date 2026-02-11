<script lang="ts">
	import type { Signal } from '$lib/types/api';
	import type { Snippet } from 'svelte';

	interface Props {
		variant: Signal | 'success' | 'error' | 'warning' | 'info' | 'neutral';
		children: Snippet;
	}

	let { variant, children }: Props = $props();

	const variantStyles: Record<string, string> = {
		BUY: 'bg-green-500/10 text-green-400 border-green-500/20',
		SELL: 'bg-red-500/10 text-red-400 border-red-500/20',
		HOLD: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
		success: 'bg-green-500/10 text-green-400 border-green-500/20',
		error: 'bg-red-500/10 text-red-400 border-red-500/20',
		warning: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
		info: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
		neutral: 'bg-slate-500/10 text-slate-400 border-slate-500/20'
	};

	const classes = $derived(variantStyles[variant] || variantStyles.neutral);
</script>

<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border {classes}">
	{@render children()}
</span>

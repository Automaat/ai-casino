<script lang="ts">
	interface Props {
		title: string;
		value: string | number;
		subtitle?: string;
		trend?: 'up' | 'down' | 'neutral';
		icon?: string;
	}

	let { title, value, subtitle, trend, icon }: Props = $props();
</script>

<div class="bg-slate-800 rounded-lg border border-slate-700 p-6">
	<div class="flex items-start justify-between">
		<div class="flex-1">
			<p class="text-sm font-medium text-slate-400">{title}</p>
			<p class="mt-2 text-3xl font-bold text-slate-100">{value}</p>
			{#if subtitle}
				<p class="mt-1 text-sm text-slate-400">{subtitle}</p>
			{/if}
		</div>
		{#if icon}
			<div class="ml-4 text-2xl text-slate-400">{icon}</div>
		{/if}
	</div>
	{#if trend}
		<div class="mt-4">
			{#if trend === 'up'}
				<span class="inline-flex items-center text-sm font-medium text-green-400">
					↑ Increasing
				</span>
			{:else if trend === 'down'}
				<span class="inline-flex items-center text-sm font-medium text-red-400">
					↓ Decreasing
				</span>
			{:else}
				<span class="inline-flex items-center text-sm font-medium text-slate-400">
					→ Stable
				</span>
			{/if}
		</div>
	{/if}
</div>

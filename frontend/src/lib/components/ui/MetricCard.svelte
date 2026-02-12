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

<div class="bg-white rounded-lg border border-gray-300 p-6">
	<div class="flex items-start justify-between">
		<div class="flex-1">
			<p class="text-sm font-medium text-gray-600">{title}</p>
			<p class="mt-2 text-3xl font-bold text-black">{value}</p>
			{#if subtitle}
				<p class="mt-1 text-sm text-gray-600">{subtitle}</p>
			{/if}
		</div>
		{#if icon}
			<div class="ml-4 text-2xl text-gray-600">{icon}</div>
		{/if}
	</div>
	{#if trend}
		<div class="mt-4">
			{#if trend === 'up'}
				<span class="inline-flex items-center text-sm font-medium text-green-600">
					↑ Increasing
				</span>
			{:else if trend === 'down'}
				<span class="inline-flex items-center text-sm font-medium text-red-600">
					↓ Decreasing
				</span>
			{:else}
				<span class="inline-flex items-center text-sm font-medium text-gray-600">
					→ Stable
				</span>
			{/if}
		</div>
	{/if}
</div>

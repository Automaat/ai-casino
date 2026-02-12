<script lang="ts">
	interface Tab {
		id: string;
		label: string;
		href: string;
	}

	interface Props {
		tabs: Tab[];
		activeTab: string;
	}

	let { tabs, activeTab }: Props = $props();
</script>

<div class="border-b border-gray-300">
	<nav class="-mb-px flex space-x-8" aria-label="Tabs">
		{#each tabs as tab}
			<a
				href={tab.href}
				class="
					{tab.id === activeTab
						? 'border-black text-black'
						: 'border-transparent text-gray-600 hover:text-black hover:border-gray-400'}
					whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
				"
				aria-current={tab.id === activeTab ? 'page' : undefined}
			>
				{tab.label}
			</a>
		{/each}
	</nav>
</div>

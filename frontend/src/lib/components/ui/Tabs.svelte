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

<div class="border-b border-slate-700">
	<nav class="-mb-px flex space-x-8" aria-label="Tabs">
		{#each tabs as tab}
			<a
				href={tab.href}
				class="
					{tab.id === activeTab
						? 'border-blue-500 text-blue-400'
						: 'border-transparent text-slate-400 hover:text-slate-300 hover:border-slate-300'}
					whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors
				"
				aria-current={tab.id === activeTab ? 'page' : undefined}
			>
				{tab.label}
			</a>
		{/each}
	</nav>
</div>

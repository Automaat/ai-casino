<script lang="ts">
	import Card from '$lib/components/ui/Card.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Accordion from '$lib/components/ui/Accordion.svelte';
	import { config } from '$lib/stores/dashboard';
	import { formatConfigValue, getSectionEnabled, sortConfigKeys } from '$lib/utils/config';
	import type * as T from '$lib/types/api';

	$: cfg = $config;

	interface SectionInfo {
		key: string;
		title: string;
	}

	interface AccordionContentData {
		category: { title: string; sections: SectionInfo[] };
		config: T.FullConfigResponse;
	}

	const categories: Array<{ title: string; sections: SectionInfo[] }> = [
		{
			title: 'Trading & Execution',
			sections: [
				{ key: 'schedule', title: 'Schedule' },
				{ key: 'analysis_orchestration', title: 'Analysis Orchestration' }
			]
		},
		{
			title: 'Risk Management',
			sections: [
				{ key: 'paper_trading', title: 'Paper Trading' },
				{ key: 'risk_limits', title: 'Risk Limits' },
				{ key: 'pre_trade_backtesting', title: 'Pre-Trade Backtesting' },
				{ key: 'position_management', title: 'Position Management' },
				{ key: 'monte_carlo', title: 'Monte Carlo' }
			]
		},
		{
			title: 'Market Surveillance',
			sections: [
				{ key: 'news_watcher', title: 'News Watcher' },
				{ key: 'social_watcher', title: 'Social Watcher' },
				{ key: 'filings_watcher', title: 'Filings Watcher' },
				{ key: 'anomaly_watcher', title: 'Anomaly Watcher' },
				{ key: 'earnings_calendar', title: 'Earnings Calendar' }
			]
		},
		{
			title: 'After-Hours Operations',
			sections: [
				{ key: 'screening', title: 'Screening' },
				{ key: 'prefetch', title: 'Prefetch' },
				{ key: 'journal', title: 'Journal' },
				{ key: 'health', title: 'Health' },
				{ key: 'optimization', title: 'Optimization' },
				{ key: 'sector_rotation', title: 'Sector Rotation' },
				{ key: 'peer_analysis', title: 'Peer Analysis' },
				{ key: 'correlation_audit', title: 'Correlation Audit' },
				{ key: 'reporting', title: 'Reporting' },
				{ key: 'rebalancing', title: 'Rebalancing' },
				{ key: 'signal_tracking', title: 'Signal Tracking' },
				{ key: 'game_plan', title: 'Game Plan' }
			]
		},
		{
			title: 'Scheduling & Infrastructure',
			sections: [
				{ key: 'state', title: 'State' },
				{ key: 'api', title: 'API' },
				{ key: 'notifications', title: 'Notifications' }
			]
		},
		{
			title: 'LLM & API Keys',
			sections: [
				{ key: 'llm', title: 'LLM' },
				{ key: 'api_keys', title: 'API Keys' }
			]
		}
	];

	$: accordionItems = cfg ? categories.map(category => ({
		id: category.title.toLowerCase().replace(/\s+/g, '-'),
		title: category.title,
		content: () => {
			return {
				category,
				config: cfg
			};
		}
	})) : [];
</script>

<svelte:head>
	<title>Config - AI Casino</title>
</svelte:head>

<div class="space-y-8">
	<div>
		<h1 class="text-3xl font-bold text-black">Configuration</h1>
		<p class="mt-2 text-gray-600">View daemon configuration (auto-refreshes every 5 seconds)</p>
	</div>

	{#if !cfg}
		<div class="text-center py-12">
			<div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600"></div>
			<p class="mt-4 text-gray-600">Loading configuration...</p>
		</div>
	{:else}
		<!-- Summary Cards -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6">
			<Card title="Watchlist">
				{#if cfg.watchlist.length > 0}
					<div class="flex flex-wrap gap-2">
						{#each cfg.watchlist as symbol}
							<Badge variant="neutral">{symbol}</Badge>
						{/each}
					</div>
				{:else}
					<p class="text-gray-600">No symbols</p>
				{/if}
			</Card>

			<Card title="Interval">
				<div class="text-3xl font-bold text-black">
					{cfg.interval_minutes}
					<span class="text-lg text-gray-600 ml-2">min</span>
				</div>
			</Card>

			<Card title="Market Hours Only">
				<div class="text-3xl">
					{#if cfg.market_hours_only}
						<span class="text-green-600">✓</span>
					{:else}
						<span class="text-red-600">✗</span>
					{/if}
				</div>
			</Card>

			<Card title="Auto Trade">
				<div class="text-3xl">
					{#if cfg.auto_trade}
						<span class="text-green-600">✓</span>
					{:else}
						<span class="text-red-600">✗</span>
					{/if}
				</div>
			</Card>

			<Card title="Trading Mode">
				<div class="text-2xl font-bold text-black uppercase">
					{cfg.trading_mode || 'UNKNOWN'}
				</div>
			</Card>

			<Card title="Pre-Market">
				<div class="text-3xl">
					{#if cfg.schedule && typeof cfg.schedule === 'object' && 'enable_pre_market' in cfg.schedule && cfg.schedule.enable_pre_market}
						<span class="text-green-600">✓</span>
					{:else}
						<span class="text-red-600">✗</span>
					{/if}
				</div>
			</Card>
		</div>

		<!-- Configuration Sections -->
		<div>
			<h2 class="text-2xl font-bold text-black mb-4">Configuration Sections</h2>
			<Accordion items={accordionItems} defaultOpen={['trading-&-execution']}>
				{#snippet content(rawData)}
					{@const data = rawData as AccordionContentData}
					<div class="space-y-4 mt-4">
						{#each data.category.sections as section}
							{@const sectionData = data.config[section.key as keyof T.FullConfigResponse]}
							{@const sectionDataRecord = typeof sectionData === 'object' && sectionData !== null && !Array.isArray(sectionData) ? sectionData as Record<string, unknown> : {}}
							{@const enabled = getSectionEnabled(sectionDataRecord)}
							{@const sortedEntries = sortConfigKeys(sectionDataRecord)}
							{@const borderColor = enabled ? 'border-l-green-700' : 'border-l-gray-300'}

							<div class="bg-gray-50 rounded-lg border border-gray-300 border-l-4 {borderColor} overflow-hidden">
								<div class="px-4 py-3 border-b border-gray-300 flex items-center justify-between">
									<h4 class="font-semibold text-black">{section.title}</h4>
									<Badge variant={enabled ? 'success' : 'neutral'}>
										{enabled ? 'Enabled' : 'Disabled'}
									</Badge>
								</div>
								<div class="overflow-x-auto">
									<table class="min-w-full divide-y divide-gray-200">
										<tbody class="divide-y divide-gray-200">
											{#each sortedEntries as [key, value]}
												{@const formatted = formatConfigValue(value)}
												{@const colorClass = formatted.color || 'text-gray-700'}

												<tr class="hover:bg-gray-50">
													<td class="px-4 py-3 text-sm text-gray-600 w-1/3">{key}</td>
													<td class="px-4 py-3 text-sm {colorClass}">
														{#if formatted.type === 'list' && Array.isArray(value) && value.length > 0}
															<div class="flex flex-wrap gap-1">
																{#each value as item}
																	<Badge variant="neutral">{String(item)}</Badge>
																{/each}
															</div>
														{:else}
															{formatted.display}
														{/if}
													</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							</div>
						{/each}
					</div>
				{/snippet}
			</Accordion>
		</div>
	{/if}
</div>

<style>
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	.animate-spin {
		animation: spin 1s linear infinite;
	}
</style>

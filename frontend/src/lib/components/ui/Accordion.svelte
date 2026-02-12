<script lang="ts">
	import type { Snippet } from 'svelte';

	interface AccordionItem {
		id: string;
		title: string;
		content: () => unknown;
	}

	interface Props {
		items: AccordionItem[];
		allowMultiple?: boolean;
		defaultOpen?: string[];
		content?: Snippet<[unknown]>;
		children?: Snippet;
	}

	let { items, allowMultiple = true, defaultOpen = [], content, children }: Props = $props();

	let openItems = $state<Set<string>>(new Set());

	$effect(() => {
		openItems = new Set(defaultOpen);
	});

	function toggle(id: string) {
		const newOpenItems = new Set(openItems);
		if (newOpenItems.has(id)) {
			newOpenItems.delete(id);
		} else {
			if (!allowMultiple) {
				newOpenItems.clear();
			}
			newOpenItems.add(id);
		}
		openItems = newOpenItems;
	}

	function isOpen(id: string): boolean {
		return openItems.has(id);
	}
</script>

<div class="space-y-3">
	{#each items as item}
		<div class="border border-gray-300 rounded-lg bg-white overflow-hidden">
			<button
				type="button"
				class="w-full px-6 py-4 text-left flex items-center justify-between hover:bg-gray-50 transition-colors"
				onclick={() => toggle(item.id)}
				aria-expanded={isOpen(item.id)}
				aria-controls={`accordion-content-${item.id}`}
			>
				<h3 class="text-lg font-semibold text-black">{item.title}</h3>
				<svg
					class="w-5 h-5 text-gray-600 transition-transform duration-200"
					class:rotate-180={isOpen(item.id)}
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
				</svg>
			</button>

			{#if isOpen(item.id)}
				<div
					id={`accordion-content-${item.id}`}
					class="px-6 pb-6 border-t border-gray-300 animate-fadeIn"
				>
					{#if content}
						{@render content(item.content())}
					{:else if children}
						{@render children()}
					{/if}
				</div>
			{/if}
		</div>
	{/each}
</div>

<style>
	@keyframes fadeIn {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	.animate-fadeIn {
		animation: fadeIn 0.2s ease-in;
	}

	.rotate-180 {
		transform: rotate(180deg);
	}
</style>

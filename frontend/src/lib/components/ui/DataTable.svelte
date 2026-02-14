<script lang="ts" generics="T">
	interface Column<T> {
		key: keyof T;
		label: string;
		format?: (value: any, row: T) => string;
		class?: string;
		cellClass?: (value: any, row: T) => string;
	}

	interface Props {
		data: T[];
		columns: Column<T>[];
		class?: string;
	}

	let { data, columns, class: className }: Props = $props();
</script>

<div class="overflow-x-auto {className}">
	<table class="min-w-full divide-y divide-gray-300">
		<thead class="bg-gray-50">
			<tr>
				{#each columns as column}
					<th
						class="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider {column.class || ''}"
					>
						{column.label}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody class="bg-white divide-y divide-gray-300">
			{#each data as row}
				<tr class="hover:bg-gray-50 transition-colors">
					{#each columns as column}
						<td class="px-6 py-4 whitespace-nowrap text-sm text-black {column.cellClass ? column.cellClass(row[column.key], row) : ''}">
							{#if column.format}
								{column.format(row[column.key], row)}
							{:else}
								{row[column.key]}
							{/if}
						</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
	{#if data.length === 0}
		<div class="text-center py-12 text-gray-600">
			No data available
		</div>
	{/if}
</div>

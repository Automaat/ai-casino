import { defineConfig } from "vitest/config";
import { sveltekit } from "@sveltejs/kit/vite";

export default defineConfig({
	plugins: [sveltekit()],
	test: {
		include: ["src/**/*.{test,spec}.{js,ts}"],
		environment: "jsdom",
		globals: true,
		coverage: {
			reporter: ["text", "json", "html"],
			exclude: ["node_modules/", "src/lib/api/"]
		}
	}
});

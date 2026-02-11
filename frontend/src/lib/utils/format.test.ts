import { describe, it, expect } from "vitest";
import {
	formatCurrency,
	formatPercent,
	formatNumber,
	formatDuration
} from "./format";

describe("formatCurrency", () => {
	it("formats positive numbers as USD currency", () => {
		expect(formatCurrency(1234.56)).toBe("$1,234.56");
	});

	it("formats negative numbers as USD currency", () => {
		expect(formatCurrency(-1234.56)).toBe("-$1,234.56");
	});

	it("formats zero as USD currency", () => {
		expect(formatCurrency(0)).toBe("$0.00");
	});

	it("formats small amounts with proper decimals", () => {
		expect(formatCurrency(0.99)).toBe("$0.99");
	});
});

describe("formatPercent", () => {
	it("formats decimal as percentage with default 2 decimals", () => {
		expect(formatPercent(0.1234)).toBe("12.34%");
	});

	it("formats decimal as percentage with custom decimals", () => {
		expect(formatPercent(0.1234, 1)).toBe("12.3%");
	});

	it("formats negative percentage", () => {
		expect(formatPercent(-0.05)).toBe("-5.00%");
	});

	it("formats zero percentage", () => {
		expect(formatPercent(0)).toBe("0.00%");
	});
});

describe("formatNumber", () => {
	it("formats number with default 0 decimals", () => {
		expect(formatNumber(1234.56)).toBe("1,235");
	});

	it("formats number with custom decimals", () => {
		expect(formatNumber(1234.56, 2)).toBe("1,234.56");
	});

	it("formats large numbers with thousand separators", () => {
		expect(formatNumber(1234567.89, 2)).toBe("1,234,567.89");
	});
});

describe("formatDuration", () => {
	it("formats seconds under 60 as seconds", () => {
		expect(formatDuration(45)).toBe("45s");
	});

	it("formats exactly 60 seconds as minutes", () => {
		expect(formatDuration(60)).toBe("1m 0s");
	});

	it("formats minutes and seconds", () => {
		expect(formatDuration(125)).toBe("2m 5s");
	});

	it("formats zero duration", () => {
		expect(formatDuration(0)).toBe("0s");
	});

	it("rounds fractional seconds", () => {
		expect(formatDuration(45.7)).toBe("46s");
	});
});

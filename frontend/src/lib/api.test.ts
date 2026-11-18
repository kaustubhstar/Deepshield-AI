import { describe, expect, it } from "vitest";
import { API_BASE } from "./api";

describe("API configuration", () => {
  it("defaults to local FastAPI during development", () => {
    expect(API_BASE).toMatch(/localhost:8000/);
  });
});

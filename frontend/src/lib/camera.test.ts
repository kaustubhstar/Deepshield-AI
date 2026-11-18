import { describe, expect, it } from "vitest";
import { permissionMessage } from "./camera";

describe("camera errors", () => {
  it("explains permission denial", () => {
    const error = new DOMException("denied", "NotAllowedError");
    expect(permissionMessage(error)).toMatch(/denied/i);
  });

  it("explains missing camera", () => {
    const error = new DOMException("missing", "NotFoundError");
    expect(permissionMessage(error)).toMatch(/No camera/i);
  });
});

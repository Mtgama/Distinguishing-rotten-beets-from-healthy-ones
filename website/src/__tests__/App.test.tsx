import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "../App";

describe("App", () => {
  it("renders the landing page on root path", () => {
    render(<App />);
    expect(
      screen.getByText(/شناسایی چغندر آفت‌زده با هوش مصنوعی/i)
    ).toBeInTheDocument();
  });

  it("renders the start button", () => {
    render(<App />);
    expect(screen.getByText(/شروع تست مدل/i)).toBeInTheDocument();
  });
});

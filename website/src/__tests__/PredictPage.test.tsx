import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import PredictPage from "../components/PredictPage";

// Mock fetch globally
vi.stubGlobal("fetch", vi.fn());

function renderWithRouter(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe("PredictPage", () => {
  it("renders the page heading", () => {
    renderWithRouter(<PredictPage />);
    expect(
      screen.getByText(/تشخیص چغندر آفت‌زده/i)
    ).toBeInTheDocument();
  });

  it("renders the upload area", () => {
    renderWithRouter(<PredictPage />);
    expect(
      screen.getByText(/کلیک کنید یا تصویر را بکشید/)
    ).toBeInTheDocument();
  });

  it("renders the back button", () => {
    renderWithRouter(<PredictPage />);
    expect(
      screen.getByText(/بازگشت به صفحه اصلی/)
    ).toBeInTheDocument();
  });

  it("has a hidden file input for image upload", () => {
    renderWithRouter(<PredictPage />);
    const input = screen.getByRole("textbox", { hidden: true }) ||
      document.querySelector('input[type="file"]');
    expect(input).toBeInTheDocument();
  });
});

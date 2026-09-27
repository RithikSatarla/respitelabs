export default function Footer() {
  const cols = [
    ["GitHub", "https://github.com/RithikSatarla/kintrace"],
    ["Demo", "/play/"],
    ["Worked incident", "/demo/"],
    ["Deck", "/pitch/"],
    ["Email", "mailto:rithiksatarla@gmail.com"],
  ];
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto flex max-w-[1240px] flex-wrap items-center justify-between gap-5 px-5 py-8 sm:px-8">
        <p className="font-mono text-[11px] text-faint">
          KINTRACE · 2026 · Results so far are from simulation.
        </p>
        <ul className="flex flex-wrap gap-5">
          {cols.map(([label, href]) => (
            <li key={label}>
              <a href={href} className="font-mono text-[11px] text-soft transition hover:text-ink">
                {label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </footer>
  );
}

const answered = {
  status: "answered",
  answer: "Admins struggle with SSO setup [atom_sso] and role mapping [atom_roles].",
  uncertainty: "The evidence covers enterprise onboarding, but not every setup path.",
  recommendations: ["Update the SSO guide [atom_sso]."],
  retrieval: {
    top_score: 0.481,
    abstain_threshold: 0.345,
  },
  citations: [
    {
      atom_id: "atom_sso",
      feedback_id: "fb_sso",
      statement: "Our admin team got stuck connecting SSO.",
    },
    {
      atom_id: "atom_roles",
      feedback_id: "fb_roles",
      statement: "The permissions page did not map clearly to enterprise roles.",
    },
  ],
};

const abstained = {
  status: "abstained",
  answer: "I do not have enough retrieved evidence to answer that reliably.",
  uncertainty: "",
  recommendations: [],
  retrieval: {
    top_score: 0.177,
    abstain_threshold: 0.345,
  },
  citations: [],
};

module.exports = { answered, abstained };
